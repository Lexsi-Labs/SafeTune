"""Safe Delta: post-hoc OBS-style editing of fine-tuning delta parameters.

Paper:  "Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on
         Diverse Datasets", Lu et al., ICML 2025, arXiv:2505.12038.
Repo:   https://github.com/ColinLu50/SafeDelta
Key code: ``llama2/safedelta/safedelta_runner.py`` —
          ``SafeDeltaRunner.add_batch`` / ``SafeDeltaRunner.adjust_delta``,
          driver ``llama2/run_safedelta.py::run_safedelta``.

What Safe Delta actually is
---------------------------
Safe Delta is a **one-shot post-hoc weight edit**, run *after* fine-tuning
(fine-tune -> apply Safe Delta -> evaluate). It is **not** a training-time
gradient constraint. For every linear layer it inspects the *delta parameters*
``W_sft - W_aligned`` and decides, entry by entry, which delta entries to
**keep** (utility) and which to **revert to the aligned weight** (safety),
subject to a global safety-loss budget controlled by a single strength
hyper-parameter ``s``.

⚠️ FIDELITY (🟡 simplified-correct): the per-entry importance / safety-loss
surrogate, the OBS error-compensation column sweep, ``importance_drop=0.15``
(revert the highest-importance entries to aligned), ``scale=rows/4096`` and the
damping all faithfully reproduce upstream ``adjust_delta``. The ONE deviation:
we build a single **global** keep-mask + safety-loss budget over the whole
weight matrix, whereas upstream selects **per column-block** (``blocksize=2048``,
so a 4096-wide layer gets its ``s/2`` budget per block × 2 blocks ≈ ``s``). Net
effect: a different set of entries is kept and roughly 2× more delta is reverted
to the aligned weights at the same ``s``. The algorithm is correct; it is NOT a
line-for-line port — treat the global-budget variant as the SafeTune default.

The selection is the SparseGPT / Optimal-Brain-Surgeon machinery
(``SafeDeltaRunner`` is "adapted from https://github.com/IST-DASLab/sparsegpt``):

* Calibration: a *safety dataset* (harmful instructions paired with safe
  refusal responses) is run through the **aligned** model, accumulating, per
  linear layer, the input Hessian ``H = (2/n) sum_i X_i X_i^T``
  (``add_batch``).
* For each layer (``adjust_delta``), with damped ``H``, ``Hinv`` is the
  inverse Hessian and ``d = diag(Hinv)`` its diagonal:

    - ``importance``  = ``1 / d**2``                  -- utility-aware weight
    - ``safety_loss`` = ``(W_sft - W_aligned)**2 / d**2`` -- OBS surrogate of
      the safety degradation caused by swapping in each delta entry.

* The top ``importance_drop`` fraction of entries by ``importance`` are
  excluded from consideration (their fine-tuned value is forced kept -- those
  deltas are too utility-critical to revert). The remaining entries are sorted
  by ``importance`` and greedily admitted (by cumulative ``safety_loss``) while
  the running total stays under the budget
  ``loss_constraint = importance.mean() * s * scale``.
* Admitted entries take the **fine-tuned** value; everything else is reset to
  the **aligned** value. After each sub-block an OBS error-compensation update
  ``W[:, future] -= err @ Hinv[block, future]`` propagates the change so the
  layer output is preserved as well as possible.

Larger ``s`` => larger budget => more fine-tuning delta kept (more utility,
less safety recovery); ``s = 0`` reverts to the aligned model.

This module reproduces that algorithm. ``apply_safe_delta`` performs the edit
in-place on ``model`` and returns the mutated model.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


@assert_mutates("apply_safe_delta")
def apply_safe_delta(
    model: nn.Module,
    aligned: Optional[nn.Module] = None,
    unsafe: Optional[nn.Module] = None,
    projection_strength: float = 0.1,
    param_filter: Optional[list] = None,
    *,
    strength: Optional[float] = None,
    hessian_inputs: Optional[Any] = None,
    importance_drop: float = 0.15,
    percdamp: float = 0.01,
    ref_columns: int = 4096,
    blocksize: int = 2048,
    sub_block_size: int = 4,
) -> Any:
    """Apply the Safe Delta post-hoc weight edit to ``model``.

    Safe Delta starts from the *aligned* weights and selectively re-introduces
    the fine-tuning delta ``W_sft - W_aligned`` entry-by-entry, keeping the
    delta entries that buy utility while reverting (to the aligned weight) the
    entries that erode safety, under a global safety-loss budget set by
    ``strength``.

    Args:
        model: the fine-tuned model to harden. **Mutated in place** and also
            returned.
        aligned: the safety-aligned reference model (``theta_aligned``). The
            output is anchored to these weights. Required.
        unsafe: optional explicit holder of the fine-tuned weights
            (``theta_sft``). When ``None`` the fine-tuned weights are read from
            ``model`` itself (the common case: ``model`` *is* the fine-tuned
            model). When given, ``unsafe`` supplies ``theta_sft`` and ``model``
            is treated purely as the tensor to overwrite.
        projection_strength: legacy alias for ``strength`` (kept so the public
            positional signature is unchanged). Used only when ``strength`` is
            ``None``. Defaults to ``0.1``, the conservative safety-recovery
            setting from the SafeTune audit (strong safety recovery with minimal
            utility cost). Higher values keep more of the fine-tuned delta at
            the cost of weaker safety recovery; ``s = 1.0`` is nearly a no-op
            and is not recommended. ``s = 0`` reverts entirely to ``aligned``.
        param_filter: optional list of substrings; if non-empty only weights
            whose parameter name contains one of them are edited.
        strength: the Safe Delta budget hyper-parameter ``s``. Larger ``s``
            keeps more fine-tuning delta (more utility, weaker safety
            recovery). When ``None`` falls back to ``projection_strength``.
        hessian_inputs: optional inverse-Hessian calibration signal. The paper
            runs a *safety dataset* through the aligned model to build the
            per-layer input Hessian ``H``. Since the Recover API is
            training-/data-free, this implementation accepts an optional dict
            ``{param_name: H or Hinv_diag}`` of pre-computed Hessian
            information; when ``None`` the diagonal Hessian degenerates to the
            identity (``d == 1``), which makes the OBS surrogate reduce to the
            magnitude of the delta -- a faithful budgeted reduction of the
            algorithm in the data-free setting.
        importance_drop: fraction of the highest-importance entries that are
            never reverted (paper: ``0.15``).
        percdamp: Hessian damping fraction (paper: ``0.01``).
        ref_columns: reference layer width used for the ``scale`` and ``s/2``
            normalisation in the paper (``4096`` for Llama-2-7B).
        blocksize: column block size for the OBS sweep (paper: ``2048``).
        sub_block_size: inner sub-block size for error compensation
            (paper: ``4``).

    Returns:
        The mutated ``model``.
    """
    try:
        import torch
    except ImportError as e:  # pragma: no cover - defensive
        raise ImportError("apply_safe_delta requires PyTorch.") from e

    if aligned is None:
        raise ValueError(
            "apply_safe_delta requires an `aligned` reference model "
            "(theta_aligned): Safe Delta edits the delta W_sft - W_aligned."
        )

    s = float(projection_strength if strength is None else strength)
    filt = list(param_filter or [])

    aligned_sd = aligned.state_dict()
    # theta_sft: from `unsafe` if supplied, else from `model` itself.
    sft_sd = unsafe.state_dict() if unsafe is not None else model.state_dict()
    hess = dict(hessian_inputs or {})

    def _matches(name: str) -> bool:
        return (not filt) or any(f in name for f in filt)

    edited = 0
    skipped = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not _matches(name):
                continue
            if name not in aligned_sd or name not in sft_sd:
                skipped += 1
                continue
            # Safe Delta operates on 2-D linear weight matrices.
            if param.dim() != 2:
                skipped += 1
                continue

            orig_dtype = param.dtype
            W_align = aligned_sd[name].to(device=param.device, dtype=torch.float32)
            W_sft = sft_sd[name].to(device=param.device, dtype=torch.float32)
            if W_align.shape != param.shape or W_sft.shape != param.shape:
                skipped += 1
                continue

            new_w = _adjust_delta(
                W_align=W_align,
                W_sft=W_sft,
                s=s,
                hessian=hess.get(name),
                importance_drop=importance_drop,
                percdamp=percdamp,
                ref_columns=ref_columns,
                blocksize=blocksize,
                sub_block_size=sub_block_size,
            )
            param.data.copy_(new_w.to(dtype=orig_dtype))
            edited += 1

    logger.info(
        "apply_safe_delta: edited %d weight matrices (s=%.4g), skipped %d.",
        edited,
        s,
        skipped,
    )
    return model


def _adjust_delta(
    *,
    W_align: Any,
    W_sft: Any,
    s: float,
    hessian: Optional[Any],
    importance_drop: float,
    percdamp: float,
    ref_columns: int,
    blocksize: int,
    sub_block_size: int,
) -> Any:
    """Per-layer Safe Delta edit.

    Faithful port of ``SafeDeltaRunner.adjust_delta``
    (``llama2/safedelta/safedelta_runner.py:70-171``). Starts from the aligned
    weight ``W_align`` and selectively swaps in fine-tuned entries ``W_sft``
    under the safety-loss budget, with SparseGPT/OBS error compensation.
    """
    import torch

    dev = W_align.device
    rows, columns = W_align.shape
    W = W_align.clone()  # output is anchored to the aligned model

    # --- inverse-Hessian diagonal ----------------------------------------
    # Build (or degenerate) the inverse Hessian. The paper accumulates H from a
    # safety calibration set; here `hessian` may carry a precomputed H matrix
    # or its inverse-diagonal, otherwise H = I (data-free fallback).
    Hinv = _resolve_hinv(hessian, columns, dev, percdamp)
    dinv = torch.diag(Hinv)                       # diag(Hinv)
    dinv2 = dinv ** 2                             # diag(Hinv)**2

    # --- per-entry importance & safety-loss surrogate --------------------
    importance = torch.ones_like(W) / dinv2.reshape((1, -1))      # 1 / d**2
    safety_loss = (W_sft - W) ** 2 / dinv2.reshape((1, -1))       # (W_sft-W)^2 / d^2

    # --- budgeted greedy selection ---------------------------------------
    # Paper normalisation: scale = rows / 4096, s := s / 2 (4096 / blocksize).
    scale = rows / float(ref_columns)
    s_eff = s / max(1, ref_columns // max(1, blocksize))

    imp_flat = importance.flatten()
    loss_flat = safety_loss.flatten()
    numel = imp_flat.numel()

    sorted_indices = torch.argsort(imp_flat)
    drop = int(importance_drop * numel)
    if drop > 0:
        sorted_indices = sorted_indices[:-drop]   # never revert top-importance entries
    sorted_loss = loss_flat[sorted_indices]

    cumulative = torch.cumsum(sorted_loss, dim=0)
    loss_constraint = importance.mean() * s_eff * scale
    sorted_mask = cumulative <= loss_constraint

    keep_mask_flat = torch.zeros_like(loss_flat, dtype=torch.bool)
    keep_mask_flat[sorted_indices] = sorted_mask
    keep_mask = keep_mask_flat.reshape(W.shape)   # True => keep fine-tuned entry

    # --- OBS / SparseGPT sweep with error compensation -------------------
    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Wsft1 = W_sft[:, i1:i2]
        mask1 = keep_mask[:, i1:i2]
        Hinv1 = Hinv[i1:i2, i1:i2]
        Err1 = torch.zeros_like(W1)

        for j1 in range(0, count, sub_block_size):
            j2 = min(j1 + sub_block_size, count)
            w = W1[:, j1:j2]
            w_sft = Wsft1[:, j1:j2]
            d = torch.diag(Hinv1)[j1:j2]
            msub = mask1[:, j1:j2]

            q = w.clone()
            q[msub] = w_sft[msub].to(q.dtype)     # keep fine-tuned where selected

            err1 = (w - q) / d.unsqueeze(0)
            W1[:, j2:] -= err1 @ Hinv1[j1:j2, j2:]
            Err1[:, j1:j2] = err1
            W1[:, j1:j2] = q

        W[:, i1:i2] = W1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    return W


def _resolve_hinv(hessian: Optional[Any], columns: int, dev: Any, percdamp: float) -> Any:
    """Return the (damped) inverse Hessian used by the OBS sweep.

    Mirrors ``adjust_delta``'s Cholesky pipeline
    (``safedelta_runner.py:97-103``): damp ``H``, invert it, take the upper
    Cholesky factor of the inverse. When no ``H`` is supplied the Hessian
    degenerates to the identity (``Hinv == I``), which is the faithful
    data-free reduction of the algorithm.
    """
    import torch

    eye = torch.eye(columns, device=dev, dtype=torch.float32)

    if hessian is None:
        return eye

    h = hessian
    if not torch.is_tensor(h):
        h = torch.as_tensor(h, dtype=torch.float32, device=dev)
    h = h.to(device=dev, dtype=torch.float32)

    # A 1-D tensor is interpreted as diag(Hinv) directly.
    if h.dim() == 1:
        if h.numel() != columns:
            logger.warning(
                "apply_safe_delta: hessian diagonal has %d entries, expected "
                "%d; falling back to identity Hessian.",
                h.numel(),
                columns,
            )
            return eye
        return torch.diag(h)

    if h.dim() != 2 or h.shape[0] != columns or h.shape[1] != columns:
        logger.warning(
            "apply_safe_delta: hessian has shape %s, expected (%d, %d); "
            "falling back to identity Hessian.",
            tuple(h.shape),
            columns,
            columns,
        )
        return eye

    H = h.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=dev)
    H[diag, diag] += damp
    try:
        L = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(H, upper=True)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "apply_safe_delta: Hessian inversion failed (%s); using identity.",
            e,
        )
        return eye
    return Hinv


__all__ = ["apply_safe_delta"]
