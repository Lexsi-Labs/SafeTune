"""
LSSF: Low-Rank Safety Subspace Fusion.

Paper
-----
"LSSF: Safety Alignment for Large Language Models through Low-Rank Safety
Subspace Fusion", Guanghao Zhou, Panjia Qiu, Cen Chen, Hongyu Li, Jason Chu,
Xin Zhang, Jun Zhou. ACL 2025 (Long Papers), pp. 30621-30638.
https://aclanthology.org/2025.acl-long.1479/
(No official code repository was released by the authors.)

What LSSF does
--------------
Given an aligned ("safe") checkpoint, an un-aligned / base checkpoint, and a
fine-tuned checkpoint that has drifted away from safety, LSSF re-aligns the
fine-tuned model with a *training-free, post-hoc* weight edit.

The safety vector is the alignment delta::

    delta_safe = theta_aligned - theta_base                       # Eq. 1

LSSF observes that safety information lives in a *low-rank* subspace that is
stable across fine-tuning and largely disjoint from task capability. It builds
an orthogonal projector ``P^(r)`` onto the top-r left-singular subspace of the
safety signal and adds the *projection* of the safety vector back into the
fine-tuned model (paper Eq. 13)::

    theta'_DST = theta_aligned + delta_DST + alpha * P^(r) @ delta_safe

Because ``theta_aligned + delta_DST`` is exactly the fine-tuned checkpoint
(``delta_DST`` is the task fine-tuning delta), this reduces in practice to::

    theta_finetuned += alpha * P^(r) @ delta_safe

i.e. only the *safety-subspace component* of the alignment delta is re-injected
-- not the full delta (RESTA) and not a bare rank-k reconstruction.

Per-layer dynamic rank
----------------------
Not every layer encodes safety at the same density. LSSF picks the rank ``r``
*per layer* with the **safety singular value entropy** (paper Eqs. 5-7). With
the squared-singular-value energy distribution ``p_i = sigma_i^2 / sum_j
sigma_j^2``::

    H_rho = -sum_{i<=rho} p_i * log(p_i)                           # Eq. 6

``r`` is the smallest rank whose cumulative entropy retains a fraction ``eta``
of the full-spectrum entropy, ``H_r / H_n >= eta`` (Eq. 7).

Weighted projection
-------------------
The faithful variant scales each retained singular direction by ``alpha_i``
(paper Eqs. 10-11), linearly interpolated by singular-value magnitude::

    alpha_i = 1 + (alpha_1 - 1) * (sigma_i - sigma_r) / (sigma_1 - sigma_r)

so dominant safety directions are emphasised. The projector becomes
``P'^(r) = U'^(r) (U'^(r))^T`` with the columns of ``U`` pre-scaled by
``alpha_i`` directly, so the symmetric ``P'^(r) = U'^(r)(U'^(r))^T`` carries
``alpha_i²`` weights — exactly as written in Eq. 10-11.

Implementation note (calibration data)
--------------------------------------
The paper constructs the singular subspace from *normalized activations*
``Z = W @ X_hat`` on a small anchor/calibration set (Eq. 4). When no
calibration data is available -- the training-free, data-free setting this
module targets -- the projector is instead built from the SVD of the safety
weight delta itself. The safety delta and its own activation Gram share their
dominant left-singular subspace, so the weight-space SVD is the data-free
surrogate for the activation SVD; it is an approximation of the paper's
activation-based subspace, documented as such. If a caller already has the
activation-derived left-singular basis it can be passed via
``subspace_basis`` and is used verbatim.

Embeddings, ``lm_head`` and 1-D params (norms, biases) are skipped: they carry
no meaningful 2-D safety subspace.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


def _safety_critical_rank(S: torch.Tensor, eta: float, max_rank: int) -> int:
    """Smallest rank whose cumulative singular-value entropy retains ``eta``.

    Implements paper Eqs. 5-7. ``S`` is the (descending) singular-value vector.
    Returns a rank in ``[1, min(max_rank, S.numel())]``.
    """
    n = int(S.numel())
    if n == 0:
        return 0
    cap = max(1, min(max_rank, n))
    # Energy distribution p_i = sigma_i^2 / sum sigma_j^2  (Eq. 5).
    energy = (S.double() ** 2)
    total = energy.sum()
    if total <= 0:
        return cap
    p = energy / total
    # Per-component entropy term; 0*log0 := 0.
    safe_p = torch.where(p > 0, p, torch.ones_like(p))
    terms = -p * torch.log(safe_p)
    H_full = terms.sum()
    if H_full <= 0:
        # Spectrum dominated by a single direction -> rank 1 suffices.
        return min(1, cap)
    H_cum = torch.cumsum(terms, dim=0)
    ratio = H_cum / H_full
    # First index where the retention threshold is met (Eq. 7).
    meets = (ratio >= eta).nonzero(as_tuple=False)
    r = int(meets[0].item()) + 1 if meets.numel() > 0 else n
    return max(1, min(r, cap))


def _weighted_basis(U: torch.Tensor, S: torch.Tensor, r: int, weight_max: float) -> torch.Tensor:
    """Top-r left-singular basis with columns scaled per paper Eqs. 10-11.

    ``alpha_i`` is interpolated by singular-value magnitude in ``[1, weight_max]``.
    Columns are scaled by ``alpha_i`` (paper Eq. 10) so ``U' U'^T`` carries ``alpha_i²``.
    """
    Ur = U[:, :r]
    if weight_max <= 1.0 or r <= 1:
        return Ur
    Sr = S[:r]
    s1, sr = Sr[0], Sr[-1]
    span = (s1 - sr)
    if span <= 0:
        return Ur
    alpha_i = 1.0 + (weight_max - 1.0) * (Sr - sr) / span  # Eq. 11
    return Ur * alpha_i.clamp_min(0.0).to(Ur.dtype)


@assert_mutates("apply_lssf")
def apply_lssf(
    finetuned: nn.Module,
    base: nn.Module,
    aligned: nn.Module,
    alpha: float = 1.0,
    rank: int = 8,
    min_param_dim: int = 4,
    skip_param_substrings: Optional[list] = None,
    eta: Optional[float] = None,
    weight_max: float = 1.0,
    subspace_basis: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    """Apply Low-Rank Safety Subspace Fusion in place to ``finetuned``.

    Re-injects the safety-subspace component of the alignment delta
    ``aligned - base`` into the fine-tuned model (paper Eq. 13, reduced form
    ``theta_finetuned += alpha * P^(r) @ delta_safe``).

    Args:
        finetuned: model to be patched (mutated in place).
        base: un-aligned / pre-alignment reference checkpoint.
        aligned: safety-aligned reference checkpoint.
        alpha: global scaling coefficient on the projected safety component
            (paper's ``alpha`` in Eq. 13; experiments use ~1.0-3.0).
        rank: upper bound on the per-layer safety-critical rank. When ``eta``
            is set this is just a cap; otherwise this fixed rank is used.
        min_param_dim: skip params smaller than this in any dimension.
        skip_param_substrings: skip params whose name contains any of these
            substrings. Default skips embeddings, ``lm_head`` and norms.
        eta: entropy-retention threshold in ``(0, 1]`` for the dynamic
            per-layer rank (paper Eqs. 5-7, ``H_r / H_n >= eta``). If ``None``,
            the fixed ``rank`` is used for every layer (legacy behaviour).
            Typical paper values are 0.8-0.9.
        weight_max: ``alpha_1`` for the weighted projection (paper Eqs. 10-11).
            ``1.0`` (default) gives the plain orthogonal projector ``U_r U_r^T``;
            values > 1 emphasise dominant safety directions.
        subspace_basis: optional dict mapping parameter name to a precomputed
            left-singular basis ``U`` (shape ``[d_out, k]``) -- e.g. derived
            from the paper's activation SVD on calibration data. When provided
            for a param, that basis is used instead of the weight-delta SVD.

    Returns:
        The patched ``finetuned`` model (mutated in place).
    """
    skip = list(skip_param_substrings or ["embed_tokens", "lm_head", "norm"])
    subspace_basis = subspace_basis or {}

    base_sd = base.state_dict()
    aligned_sd = aligned.state_dict()
    ft_sd = finetuned.state_dict()

    edited = 0
    skipped = 0
    rank_sum = 0
    with torch.no_grad():
        for name, ft_w in ft_sd.items():
            if any(s in name for s in skip):
                skipped += 1
                continue
            if name not in base_sd or name not in aligned_sd:
                continue
            if ft_w.dim() != 2:
                skipped += 1
                continue
            if min(ft_w.shape) < min_param_dim:
                skipped += 1
                continue

            # Safety vector delta_safe = theta_aligned - theta_base  (Eq. 1).
            w_aligned_device = aligned_sd[name].to(device=ft_w.device, dtype=ft_w.dtype)
            w_base_device = base_sd[name].to(device=ft_w.device, dtype=ft_w.dtype)
            
            delta = (w_aligned_device - w_base_device).float()

            # Build the left-singular basis of the safety subspace.
            if name in subspace_basis:
                # Caller-supplied basis (e.g. paper's activation-SVD U).
                U = subspace_basis[name].to(delta.device).float()
                S = None
            else:
                # Data-free surrogate: SVD of the safety weight delta itself.
                try:
                    U, S, _ = torch.linalg.svd(delta, full_matrices=False)
                except RuntimeError as e:
                    logger.warning("LSSF: SVD failed for %s (%s); skipping.", name, e)
                    continue

            # Per-layer safety-critical rank.
            if eta is not None and S is not None:
                r = _safety_critical_rank(S, eta=float(eta), max_rank=rank)
            else:
                k_avail = U.shape[1]
                r = max(1, min(rank, k_avail))
            if r == 0:
                skipped += 1
                continue

            # Weighted orthogonal projector P^(r) = U'_r (U'_r)^T  (Eqs. 8-11).
            if S is not None:
                Ur = _weighted_basis(U, S, r, weight_max)
            else:
                Ur = U[:, :r]

            # Project the safety vector onto the safety subspace and re-inject:
            # theta_finetuned += alpha * P^(r) @ delta_safe   (Eq. 13).
            proj = Ur @ (Ur.transpose(0, 1) @ delta)
            ft_w.add_((alpha * proj).to(ft_w.dtype))
            edited += 1
            rank_sum += r

    avg_rank = (rank_sum / edited) if edited else 0.0
    logger.info(
        "LSSF: edited %d 2-D params (skipped %d), alpha=%.2f eta=%s avg_rank=%.1f.",
        edited, skipped, alpha, eta, avg_rank,
    )
    return finetuned


__all__ = ["apply_lssf"]
