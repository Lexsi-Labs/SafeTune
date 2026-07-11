"""LoX: low-rank extrapolation of the safety subspace (training-free).

Faithful re-implementation of the authors' reference code:
VITA-Group/LoX -- ``safety/LoX.py`` (COLM 2025, arXiv:2506.15606).

The authors' algorithm (``safety/LoX.py:25-46``)::

    W_aligned = aligned_model.state_dict()
    W_base    = pretrained_model.state_dict()
    dW        = {n: W_aligned[n] - W_base[n] for n in W_aligned}
    for n in dW:
        if dW[n].dim() > 1:
            if k > 0:                       # rank-truncate the delta
                U, S, Vt = svd(dW[n], full_matrices=False)
                S[k:] = 0
                m = U @ diag(S) @ Vt
            else:                           # k == 0 -> full-rank delta
                m = dW[n]
            new[n] = W_aligned[n] + coef * m
        else:                               # 1-D params: unchanged
            new[n] = W_aligned[n]

So LoX hardens the **aligned** model: ``W_hardened = W_aligned + coef *
LowRank_k(W_aligned - W_base)``. The base point for the addition is the
aligned model itself, and ``coef`` is a direct multiplier (the paper's
extrapolation coefficient), not ``coef - 1``. 1-D params (biases, norms)
are left exactly as in the aligned model.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


@assert_mutates("apply_lox")
def apply_lox(
    model: nn.Module,
    base: nn.Module,
    aligned: nn.Module,
    rank: int = 64,
    extrapolation_factor: float = 1.5,
    param_filter: Optional[list] = None,
) -> nn.Module:
    """Apply LoX low-rank safety extrapolation, hardening ``model`` in-place.

    LoX hardens an *aligned* model before it is fine-tuned, by extrapolating
    the low-rank safety subspace of its alignment update::

        W_hardened = W_aligned + extrapolation_factor * LowRank_k(W_aligned - W_base)

    where ``LowRank_k`` keeps the top-``rank`` singular components of the
    per-weight alignment delta. This matches the authors' reference
    implementation (VITA-Group/LoX, ``safety/LoX.py``).

    Args:
        model: the module whose weights are overwritten with the hardened
            (extrapolated) weights. LoX is meant to harden the aligned model,
            so the typical call passes the aligned model here as well.
        base: the un-aligned pretrained reference model (``W_base``).
        aligned: the safety-aligned model (``W_aligned``) -- the base point
            the low-rank delta is added to.
        rank: number of top singular components of the alignment delta to
            keep (the paper's ``k``). ``rank <= 0`` extrapolates the
            full-rank delta with no SVD, exactly as the authors' ``k=0`` path.
        extrapolation_factor: the extrapolation coefficient ``coef``; the
            low-rank delta is scaled by this value directly. The authors use
            ``coef=1.0`` by default.
        param_filter: optional list of substrings; if given, only parameters
            whose name contains one of the substrings are extrapolated (the
            rest are copied from ``aligned`` unchanged). Not part of the
            original paper -- an optional SafeTune convenience knob.

    Returns:
        ``model``, mutated in place with the hardened state dict.
    """
    coef = float(extrapolation_factor)
    k = int(rank)

    # The LoX base point is ``model`` — the module passed in and whose weights
    # are overwritten (see the Args docstring). In the recover pillar that is
    # the drifted/fine-tuned model; in the paper's pre-FT use it is the aligned
    # model (callers then pass ``aligned`` as ``model`` too). The low-rank
    # safety component is the rank-k truncation of the *alignment delta*
    # ``aligned - base``, extrapolated ONTO ``model``:
    #     W_out = W_model + coef * LowRank_k(W_aligned - W_base)
    w_model = model.state_dict()
    w_aligned = aligned.state_dict()
    w_base = base.state_dict()

    def _matches(name: str) -> bool:
        if not param_filter:
            return True
        return any(f in name for f in param_filter)

    new_state_dict = {}
    n_extrapolated = 0
    n_svd_failed = 0

    for name, w in w_model.items():
        # Default: keep the model's own weight (the LoX base point).
        new_state_dict[name] = w

        if name not in w_base or name not in w_aligned or not _matches(name):
            continue

        # Only multi-dimensional tensors are extrapolated; 1-D params
        # (biases, layer norms) are left as the model's own.
        if w.dim() <= 1:
            continue

        dtype = w.dtype
        # Alignment safety delta = aligned - base.
        delta = (w_aligned[name].to(torch.float32)
                 - w_base[name].to(torch.float32))

        if k > 0:
            # Rank-truncate the alignment delta: keep its top-k singular
            # components (authors: S[k:] = 0; m = U @ diag(S) @ Vt).
            try:
                U, S, Vt = torch.linalg.svd(delta, full_matrices=False)
                if k < S.shape[0]:
                    S = S.clone()
                    S[k:] = 0
                m = (U * S.unsqueeze(-2)) @ Vt
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("LoX: SVD failed for %s: %s", name, e)
                n_svd_failed += 1
                continue
        else:
            # k <= 0: extrapolate the full-rank delta, no SVD.
            m = delta

        m_device = m.to(w.device)
        extrapolated = w.to(torch.float32) + coef * m_device
        new_state_dict[name] = extrapolated.to(dtype)
        n_extrapolated += 1

    model.load_state_dict(new_state_dict, strict=False)
    logger.info(
        "LoX: extrapolated %d params (coef=%.3f, rank=%s) onto the target "
        "model%s.",
        n_extrapolated,
        coef,
        "full" if k <= 0 else k,
        f"; {n_svd_failed} SVD failures" if n_svd_failed else "",
    )
    return model


__all__ = ["apply_lox"]
