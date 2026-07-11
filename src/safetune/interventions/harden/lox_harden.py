"""
LoX Harden — Training-free Safety-Subspace Extrapolation (Perin et al., 2025,
arXiv:2506.15606, COLM 2025).

LoX (Low-rank eXtrapolation) strengthens the safety subspace of an aligned model
BEFORE fine-tuning begins, making the model more robust to subsequent LoRA attacks.

Algorithm:
1. Compute the alignment delta: delta_W = W_aligned - W_base
2. For each weight matrix W_aligned, take the SVD of delta_W:
   delta_W = U @ S @ V^T
3. Extrapolate along the top safety singular directions:
   W_lox = W_aligned + alpha * U[:, :k] @ diag(S[:k]) @ V[:, :k]^T
   = W_aligned + alpha * delta_W_low_rank
4. The resulting W_lox has an amplified safety subspace.

This is a PRE-FT intervention: apply LoX to the base model before fine-tuning
begins, then fine-tune the resulting model with a standard LoRA trainer.
The amplified safety subspace is harder for the fine-tuning to erase.

Paper defaults: alpha=1.0, rank=None (full rank of delta_W).

Note: This is a harden method because it is applied BEFORE fine-tuning to make
the model more resistant to drift. It differs from the Recover pillar's apply_lox
(which is applied POST fine-tuning to restore safety). Both use the same SVD
extrapolation math but at different points in the pipeline.

Reference: Perin et al. "LoX: Safety through Low-Rank Extrapolation" COLM 2025.
Official code: https://github.com/VITA-Group/LoX
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LoXHardenConfig:
    """Configuration for LoX pre-FT safety subspace extrapolation.

    Attributes:
        alpha: Extrapolation strength. alpha=1.0 doubles the safety subspace
            magnitude; alpha=0.5 amplifies by 50%. Paper default: 1.0.
            Higher values give stronger protection but may hurt capability.
        rank: Number of singular vectors to use. None = all non-zero singular
            values (full low-rank structure of the alignment delta).
        param_filter: Optional list of substrings; only matching parameter names
            are extrapolated. None = all weight matrices with a 2D+ shape.
            Example: ["q_proj", "v_proj"] to target attention only.
        min_singular_value: Threshold below which singular values are treated
            as zero (numerical noise). Default: 1e-6.
    """
    alpha: float = 1.0
    rank: Optional[int] = None
    param_filter: Optional[list] = None
    min_singular_value: float = 1e-6


def apply_lox_harden(
    model: nn.Module,
    base_state_dict: Dict[str, torch.Tensor],
    aligned_state_dict: Dict[str, torch.Tensor],
    config: Optional[LoXHardenConfig] = None,
) -> nn.Module:
    """Apply LoX extrapolation to amplify the safety subspace before fine-tuning.

    Modifies model weights IN-PLACE to amplify the alignment direction.

    Args:
        model: The aligned model to be hardened (will be fine-tuned afterwards).
            Modified in-place.
        base_state_dict: State dict of the base (unaligned) model.
        aligned_state_dict: State dict of the aligned model (same architecture).
        config: LoXHardenConfig.

    Returns:
        The same model with amplified safety subspace (modified in-place).
    """
    cfg = config or LoXHardenConfig()
    n_applied = 0
    n_skipped = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                n_skipped += 1
                continue

            if cfg.param_filter is not None:
                if not any(s in name for s in cfg.param_filter):
                    n_skipped += 1
                    continue

            w_base = base_state_dict.get(name)
            w_aligned = aligned_state_dict.get(name)

            if w_base is None or w_aligned is None:
                logger.debug("LoX harden: skipping %s (not in both state dicts)", name)
                n_skipped += 1
                continue

            if w_base.shape != w_aligned.shape:
                logger.debug("LoX harden: skipping %s (shape mismatch)", name)
                n_skipped += 1
                continue

            # Alignment delta in float32 for numerical stability.
            delta = (w_aligned - w_base).float()

            # Flatten to 2D for SVD: (out_dim, in_dim).
            orig_shape = delta.shape
            if delta.dim() > 2:
                delta_2d = delta.reshape(delta.shape[0], -1)
            else:
                delta_2d = delta

            # SVD of the alignment delta.
            try:
                U, S, Vh = torch.linalg.svd(delta_2d, full_matrices=False)
            except Exception as exc:
                logger.warning("LoX harden: SVD failed for %s (%s); skipping.", name, exc)
                n_skipped += 1
                continue

            # Filter near-zero singular values.
            mask = S > cfg.min_singular_value
            if not mask.any():
                n_skipped += 1
                continue

            # Apply rank truncation if requested.
            if cfg.rank is not None:
                k = min(cfg.rank, mask.sum().item())
                mask[k:] = False

            U_k = U[:, mask]
            S_k = S[mask]
            Vh_k = Vh[mask, :]

            # Extrapolated delta (low-rank safety amplification).
            delta_lox = U_k @ torch.diag(S_k) @ Vh_k  # (out_dim, in_dim)

            if delta_2d.shape != delta.shape:
                delta_lox = delta_lox.reshape(orig_shape)

            # W_lox = W_aligned + alpha * delta_low_rank
            extrapolated = param.float() + cfg.alpha * delta_lox.to(param.device)
            param.data.copy_(extrapolated.to(param.dtype))
            n_applied += 1

    logger.info(
        "LoX harden: applied to %d parameters, skipped %d.",
        n_applied, n_skipped,
    )
    return model


__all__ = ["LoXHardenConfig", "apply_lox_harden"]