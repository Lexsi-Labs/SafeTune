"""WiSE-FT safety recovery (arXiv:2412.19512, Dec 2024).

WiSE-FT (Weight-Space Ensembling for Fine-Tuning, Wortsman et al., 2022)
interpolates the weights of two checkpoints to recover the best of both:

    theta_wise = alpha * theta_pre + (1 - alpha) * theta_post

Applied to safety recovery (arXiv:2412.19512): ``aligned`` is the
safety-preserving anchor (the "pre-FT" WiSE endpoint) and ``model``
(the drifted fine-tune) is the "post-FT" endpoint:

    theta_wise = alpha * theta_aligned + (1 - alpha) * theta_drifted

``alpha = 0`` keeps the fully drifted model; ``alpha = 1`` fully restores
the aligned reference. Intermediate values trade off safety recovery against
task capability retention.

Data-free — no calibration set is required. All compute is pure weight
arithmetic on the CPU.

This differs from ``apply_prepost_merge`` (which interpolates toward the
BASE / pre-fine-tune checkpoint) in that it interpolates directly toward
the ALIGNED safety reference, making it a pure safety-axis interpolation.

vLLM backend
------------
No inference is required; the method is pure weight arithmetic.
Evaluation of the resulting checkpoint uses the vLLM backend via the
standard ``_eval_ckpt`` / ``H.eval_utility(..., backend="vllm")`` pipeline
in ``run_recover.py``.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


@assert_mutates("apply_wise_ft")
def apply_wise_ft(
    model: nn.Module,
    aligned: nn.Module,
    alpha: float = 0.5,
    param_filter: Optional[Callable[[str, torch.Tensor], bool]] = None,
) -> nn.Module:
    """WiSE-FT: interpolate drifted weights toward the aligned safety reference.

    Computes  ``W_new = alpha * W_aligned + (1 - alpha) * W_drifted``  for
    every parameter pair that exists in both ``model`` and ``aligned``.
    Mutates ``model`` in place and returns it.

    Parameters
    ----------
    model:
        The drifted / fine-tuned model to patch (mutated in-place).
    aligned:
        The safety-aligned reference checkpoint (the WiSE-FT pre-FT anchor).
    alpha:
        Interpolation coefficient towards ``aligned`` ∈ [0, 1].
        ``alpha = 0`` → keep drifted; ``alpha = 1`` → fully restore aligned.
        Default: 0.5 (equal blend, the WiSE-FT paper's recommended starting
        point for safety–utility trade-off).
    param_filter:
        Optional callable ``(name, tensor) → bool``. When supplied, only
        parameters for which the callable returns True are interpolated.

    Returns
    -------
    The mutated ``model``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    aligned_sd = aligned.state_dict()
    n_applied = 0
    n_skipped_filter = 0
    n_missing = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in aligned_sd:
                n_missing += 1
                continue
            if param_filter is not None and not param_filter(name, param):
                n_skipped_filter += 1
                continue

            aligned_p = aligned_sd[name].to(param.device, dtype=torch.float32)
            drifted_f = param.float()
            # WiSE-FT formula: alpha * pre + (1 - alpha) * post
            merged = alpha * aligned_p + (1.0 - alpha) * drifted_f
            param.copy_(merged.to(param.dtype))
            n_applied += 1

    logger.info(
        "apply_wise_ft: interpolated %d params (alpha=%.3f toward aligned; "
        "skipped_filter=%d, missing_in_aligned=%d).",
        n_applied, alpha, n_skipped_filter, n_missing,
    )
    return model


__all__ = ["apply_wise_ft"]
