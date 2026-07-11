"""Safety-vector restore: targeted task arithmetic on the harm direction.

⚠️ PROVENANCE: this is a **SafeTune-original** low-rank task-arithmetic variant,
NOT a faithful reproduction of a published method. An earlier docstring cited
"Yang et al. 2025"; no paper matching this exact formulation (anchor at the
*drifted* endpoint, ``v = aligned − drifted``, with a truncated-SVD
reconstruction of ``v``) could be located, so that citation has been removed.
It is related to — but distinct from — RESTA (``v = aligned − base``, no SVD)
and LSSF (anchors at base, projects the alignment delta onto a left-singular
subspace; see :mod:`safetune.recover.lssf`). Treat as a SafeTune heuristic.

Algorithm (SafeTune-original — targeted task arithmetic)
--------------------------------------------------------
1. Construct the *safety vector*  v_l = W_aligned - W_drifted  for each
   weight matrix.  This differs from RESTA's  v = aligned - base: here the
   vector is anchored at the *drifted* endpoint, so it captures exactly the
   harmful drift to undo.

2. For each weight matrix, retain only the top-``rank`` right singular
   directions of v_l (truncated SVD):

       v_l = U_l Σ_l V_l^T  →  v̂_l = U_l[:,:r] Σ_l[:r,:r] V_l[:r,:]^T

   This concentrates the edit on the subspace most responsible for the
   safety degradation and avoids spurious changes to capability-encoding
   directions that are orthogonal to the harm drift.

3. Apply the projected safety vector with scaling coefficient ``alpha``:

       W_new = W_drifted + alpha * v̂_l

Keyword arguments are the same as RESTA/task_arithmetic for drop-in
compatibility.  When ``rank`` is set to ``None`` (or exceeds the matrix
rank) the full safety vector is applied without truncation, reducing to
plain task arithmetic with ``v = aligned - drifted``.

vLLM backend
------------
No inference is required; the method is pure weight arithmetic (SVD +
parameter update). Evaluation uses the vLLM backend via
``_eval_ckpt`` / ``H.eval_utility(..., backend="vllm")``.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


def _truncated_svd_project(delta: torch.Tensor, rank: int) -> torch.Tensor:
    """Project ``delta`` onto its top-``rank`` right singular subspace.

    For a 2-D weight matrix delta of shape (d_out, d_in):
        U, S, Vh = svd(delta)   # Vh shape (d_in, d_in)
        projected = U[:, :r] @ diag(S[:r]) @ Vh[:r, :]

    For 1-D (bias/embedding row) tensors, the projection is skipped and the
    original delta is returned unchanged.
    """
    if delta.dim() < 2:
        return delta
    d_out, d_in = delta.shape[0], delta.shape[1]
    eff_rank = min(rank, d_out, d_in)
    if eff_rank <= 0:
        return torch.zeros_like(delta)
    try:
        # torch.linalg.svd: full_matrices=False gives economy SVD
        U, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
        # U: (d_out, k),  S: (k,),  Vh: (k, d_in)  where k = min(d_out, d_in)
        U_r = U[:, :eff_rank]
        S_r = S[:eff_rank]
        Vh_r = Vh[:eff_rank, :]
        projected = (U_r * S_r.unsqueeze(0)) @ Vh_r
    except Exception as e:  # pragma: no cover - defensive (degenerate matrix)
        logger.debug("safety_vector_restore: SVD failed for delta %s — using full delta: %s",
                     delta.shape, e)
        projected = delta.float()
    return projected.to(delta.dtype)


@assert_mutates("apply_safety_vector_restore")
def apply_safety_vector_restore(
    model: nn.Module,
    aligned: nn.Module,
    alpha: float = 1.0,
    rank: Optional[int] = 8,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Re-inject the (aligned − drifted) safety vector, projected to top-rank.

    Parameters
    ----------
    model:
        The drifted model to restore (mutated in-place).
    aligned:
        The safety-aligned reference model.
    alpha:
        Scaling coefficient for the re-injected safety vector (default 1.0
        for a full rollback along the projected harm direction).
    rank:
        Number of singular directions to keep per weight matrix. Lower rank
        yields a sparser edit concentrated on the dominant harm directions.
        ``None`` applies the full (unprojected) safety vector — equivalent to
        task arithmetic with v = aligned − drifted.
    target_modules:
        Substrings of parameter names to target. Default: all parameters with
        at least 2 dimensions (Linear weight matrices; biases are updated
        without SVD projection).

    Returns
    -------
    The mutated ``model``.
    """
    if alpha == 0.0:
        logger.warning("apply_safety_vector_restore: alpha=0 — no change applied.")
        return model

    aligned_sd = aligned.state_dict()
    n_applied = 0
    n_missing = 0

    # When no explicit target_modules are given, skip embed_tokens / lm_head /
    # norm (and other 1-D tensors are SVD-skipped anyway). This mirrors
    # apply_lssf's default skip-list so a full-rank alpha=1 edit cannot
    # overwrite embeddings/LM-head and destroy capability.
    default_skip = ["embed_tokens", "lm_head", "norm"]

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in aligned_sd:
                n_missing += 1
                continue
            if target_modules is not None:
                if not any(t in name for t in target_modules):
                    continue
            elif any(s in name for s in default_skip):
                continue

            orig_dtype = param.dtype
            drifted_f = param.float()
            aligned_f = aligned_sd[name].to(param.device, dtype=torch.float32)

            # Safety vector: how far the drifted model is from the aligned ref.
            v = aligned_f - drifted_f

            if rank is not None and param.dim() >= 2:
                v_proj = _truncated_svd_project(v, rank)
            else:
                v_proj = v

            new_val = drifted_f + alpha * v_proj.to(drifted_f.dtype)
            param.copy_(new_val.to(orig_dtype))
            n_applied += 1

    logger.info(
        "apply_safety_vector_restore: applied to %d params "
        "(alpha=%.3f, rank=%s, missing=%d).",
        n_applied, alpha, rank, n_missing,
    )
    return model


__all__ = ["apply_safety_vector_restore"]
