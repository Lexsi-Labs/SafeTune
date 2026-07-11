"""SafeMERGE: preserving safety alignment via selective layer-wise model merging.

Faithful re-implementation of SafeMERGE (Djuhera et al., ICLR 2025 Workshop,
arXiv:2503.17239; official code: https://github.com/aladinD/SafeMERGE).

Algorithm (traced to ``utils.py`` / ``get_safemerge_model.py`` in the authors'
repo):

1. Build, per weight matrix ``i``, the safety-aligned subspace projection
   matrix from the aligned-minus-base delta::

       V^i  = W_aligned^i - W_base^i
       C^i  = V^i V^iᵀ / ‖V^i‖_F          # Safe-LoRA style projection matrix

   (``compute_safelora_projection_matrices``: ``C = vec @ vec.T / fro_norm``.)

2. For each fine-tuned layer, take its update ``ΔW_f^i = W_ft^i - W_base^i``,
   project it through ``C^i`` and measure the cosine similarity between the
   raw update and its projection onto the safety subspace::

       ρ^i = cos( ΔW_f^i ,  C^i ΔW_f^i )

   (``compute_safelora_cos``: cosine of flattened ``P @ ... `` vs original.)

3. A layer is *unsafe* when ``ρ^i < τ``; only unsafe layers are **merged**
   (linearly interpolated) with the safe model — safe layers keep the
   fine-tuned weights untouched. The merge for unsafe layers is, on the
   *deltas*::

       ΔW_merge^i = (1-α)·ΔW_f^i + α·ΔW_s^i ,   ΔW_s^i = W_aligned^i - W_base^i

   which, added back onto ``W_base^i``, equals the weight-space form
   ``(1-α)·W_ft^i + α·W_aligned^i``. ``"dare_linear"`` additionally drops a
   fraction ``1-density`` of the safe delta entries and rescales, matching the
   paper's DARE merging ablation.

Non-2D parameters (biases, norms) carry no LoRA-style update and are excluded
from the projection criterion, exactly as the authors do for Qwen models.

Public signature in ``recover/__init__.py`` is preserved; new knobs are
optional keyword arguments with paper-default values.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)

# Paper default cosine-similarity threshold (get_safemerge_model.py: --cos_threshold 0.35).
_PAPER_THRESHOLD = 0.35


def _projection_cosine(delta_safety: torch.Tensor, delta_task: torch.Tensor) -> Optional[float]:
    """SafeMERGE alignment measure ρ = cos(ΔW_f, C·ΔW_f)."""
    if delta_safety.dim() != 2 or delta_task.dim() != 2:
        return None
    if delta_task.numel() == 0 or delta_safety.numel() == 0:
        return None

    v = delta_safety.float()

    # Per SafeMERGE / Safe-LoRA (and consistent with recover/safe_lora.py), the
    # projection uses the (NON-squared) Frobenius norm: C = V·Vᵀ / ‖V‖_F.
    # (Only the proj↔dw cosine below is consumed here, and cosine is
    # scale-invariant, so this divisor doesn't change layer selection — but it
    # must match the paper/Safe-LoRA if C is ever reused unnormalised.)
    fro = torch.sqrt(torch.sum(v ** 2))

    if fro < 1e-9:
        return None

    dw = delta_task.float()
    # V (Vᵀ ΔW_f) / ‖V‖_F
    proj = torch.mm(v, torch.mm(v.t(), dw)) / fro
    
    if torch.norm(proj) < 1e-12 or torch.norm(dw) < 1e-12:
        return None
        
    cos_val = F.cosine_similarity(proj.reshape(1, -1), dw.reshape(1, -1), dim=1)
    return float(cos_val.item())


def _dare_drop(delta: torch.Tensor, density: float) -> torch.Tensor:
    """DARE drop-and-rescale used by the ``dare_linear`` merge variant."""
    if density >= 1.0:
        return delta
    density = max(min(density, 1.0), 0.0)
    if density <= 0.0:
        return torch.zeros_like(delta)
        
    # FIX 3: More memory efficient mask generation
    mask = (torch.rand_like(delta, dtype=torch.float32) < density).to(delta.dtype)
    return (delta * mask) / density


@assert_mutates("apply_safemerge")
def apply_safemerge(
    finetuned: nn.Module,
    base: nn.Module,
    aligned: nn.Module,
    threshold: float = _PAPER_THRESHOLD,
    alpha: float = 0.5,
    merge_type: str = "linear",
    density: Optional[float] = None,
    only_2d: bool = True,
) -> nn.Module:
    """Selectively merge unsafe fine-tuned layers toward the safety-aligned model."""
    if merge_type not in ("linear", "dare_linear"):
        raise ValueError(
            f"merge_type must be 'linear' or 'dare_linear', got {merge_type!r}"
        )
    if merge_type == "dare_linear" and density is None:
        density = 0.5

    base_sd = base.state_dict()
    aligned_sd = aligned.state_dict()
    ft_params = dict(finetuned.named_parameters()) # Use parameters for in-place copy

    merged, considered = 0, 0
    for key, param_ft in ft_params.items():
        if key not in base_sd or key not in aligned_sd:
            continue
            
        w_base = base_sd[key]
        w_aligned = aligned_sd[key]
        
        # Access the underlying tensor data
        w_ft = param_ft.data 
        
        if w_ft.shape != w_base.shape or w_ft.shape != w_aligned.shape:
            continue

        if only_2d and w_ft.dim() != 2:
            continue

        # Calculate in fp32
        w_base_device = w_base.to(w_ft.device)
        w_aligned_device = w_aligned.to(w_ft.device) # Move aligned weights to GPU
        
        delta_task = w_ft.float() - w_base_device.float()
        delta_safety = w_aligned_device.float() - w_base_device.float() # Define delta_safety

        rho = _projection_cosine(delta_safety, delta_task)
        if rho is None:
            continue
            
        considered += 1

        if rho < threshold:
            # FIX 2: Mutate the finetuned parameter strictly in-place.
            # No need to hold a massive `new_sd` dictionary in memory.
            safe_delta = delta_safety
            if merge_type == "dare_linear":
                safe_delta = _dare_drop(delta_safety, float(density))
                
            merged_delta = (1.0 - alpha) * delta_task + alpha * safe_delta
            # Use w_base_device to prevent another CPU/GPU crash!
            merged_w = w_base_device.float() + merged_delta
            
            # Copy back to original parameter memory buffer
            param_ft.data.copy_(merged_w.to(w_ft.dtype))
            merged += 1
            
        # Free memory of intermediate tensors explicitly if working on GPU
        del delta_task, delta_safety

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "apply_safemerge: merged %d / %d eligible layers (threshold=%.3f, "
        "alpha=%.3f, merge_type=%s)",
        merged,
        considered,
        threshold,
        alpha,
        merge_type,
    )
    return finetuned

__all__ = ["apply_safemerge"]
