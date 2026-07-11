"""
T-Vaccine (Liu et al., 2024, arXiv:2410.09760).

T-Vaccine is a memory-efficient variant of Vaccine (Huang et al., NeurIPS 2024).
The key difference: instead of perturbing ALL attention layers uniformly, T-Vaccine
scores each layer by its gradient norm on a safety batch, selects only the
top-k safety-critical layers, and applies the SAM-style hidden-state perturbation
only to those layers.

Algorithm (per the paper):
1. Forward + backward on the alignment loss; collect per-layer gradient norms
   of the attention output hidden states.
2. Rank layers by gradient norm; select top-k (default k=50% of layers).
3. Apply the Vaccine perturbation ONLY to the selected layers:
   e_r = grad * rho / global_norm_of_selected_grads
4. Second forward pass with the perturbation hooks active on selected layers only.
5. Minimise the perturbed loss (pure SAM objective, no clean-loss blend).

This directly reuses the vaccine_loss primitive but restricts perturbation to
the top-k safety-critical layers identified by gradient-norm scoring.

Reference implementation: the paper does not have an official public repo, but
the algorithm is a strict subset of Vaccine with layer selection added.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .vaccine import (
    VaccineConfig,
    _get_attention_modules,
    _hidden_from_output,
)

logger = logging.getLogger(__name__)


@dataclass
class TVaccineConfig:
    """Configuration for T-Vaccine (layer-selective Vaccine).

    Attributes:
        rho: L2 perturbation budget. Same as VaccineConfig.rho. Default 2.0.
        top_k_ratio: Fraction of attention layers to perturb, ranked by
            gradient norm. 0.5 = top 50% (paper default). Range (0, 1].
        attn_module_names: Same as VaccineConfig.attn_module_names.
        alpha: blend weight (1.0 = pure perturbed loss, paper default).
    """
    rho: float = 2.0
    top_k_ratio: float = 0.5
    attn_module_names: Optional[List[str]] = None
    alpha: float = 1.0


def tvaccine_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    task_loss_fn: Callable[[Any, Dict[str, torch.Tensor]], torch.Tensor],
    config: Optional[TVaccineConfig] = None,
) -> torch.Tensor:
    """Compute the T-Vaccine layer-selective perturbation loss.

    Faithful to Liu et al. 2024: only top-k attention layers (by gradient norm)
    receive the SAM-style hidden-state perturbation.
    """
    cfg = config or TVaccineConfig()

    attn_modules = _get_attention_modules(model, cfg.attn_module_names, None)
    if not attn_modules:
        raise RuntimeError(
            "tvaccine_loss: could not locate any transformer attention modules. "
            "Pass TVaccineConfig.attn_module_names with the attention class-name "
            "substring(s) for this architecture."
        )

    # ------------------------------------------------------------------
    # Pass 1: collect hidden states and their gradients for each layer.
    # ------------------------------------------------------------------
    captured_outputs: Dict[nn.Module, torch.Tensor] = {}
    fwd_handles = []

    def _capture_hook(module: nn.Module, inp: Any, output: Any):
        hidden = _hidden_from_output(output)
        hidden.retain_grad()
        captured_outputs[module] = hidden
        return output

    for module in attn_modules:
        fwd_handles.append(module.register_forward_hook(_capture_hook))

    try:
        clean_loss = task_loss_fn(model, batch)
        ordered = [m for m in attn_modules if m in captured_outputs]
        hiddens = [captured_outputs[m] for m in ordered]
        grads = torch.autograd.grad(
            clean_loss, hiddens, retain_graph=False, allow_unused=True
        )
    finally:
        for h in fwd_handles:
            h.remove()

    # ------------------------------------------------------------------
    # Score each layer by gradient norm; select top-k.
    # ------------------------------------------------------------------
    layer_scores: List[tuple] = []
    for module, grad in zip(ordered, grads):
        if grad is not None:
            score = grad.detach().norm(p=2).item()
            layer_scores.append((score, module, grad.detach()))

    if not layer_scores:
        logger.warning("tvaccine_loss: no hidden-state gradients captured; "
                       "falling back to clean loss.")
        return task_loss_fn(model, batch)

    # Sort descending by gradient norm; keep top-k.
    layer_scores.sort(key=lambda x: x[0], reverse=True)
    k = max(1, int(len(layer_scores) * cfg.top_k_ratio))
    selected = layer_scores[:k]
    logger.debug("T-Vaccine: selected %d / %d layers by gradient norm.", k, len(layer_scores))

    # ------------------------------------------------------------------
    # Compute global norm over selected layers only; build perturbations.
    # ------------------------------------------------------------------
    selected_grads = {m: g for _, m, g in selected}
    global_norm = torch.norm(
        torch.stack([g.norm(p=2) for g in selected_grads.values()]), p=2
    )
    scale = cfg.rho / (global_norm + 1e-7)
    perturbations = {m: g * scale for m, g in selected_grads.items()}

    # ------------------------------------------------------------------
    # Pass 2: perturbed forward on selected layers only.
    # ------------------------------------------------------------------
    pert_handles = []

    def _make_perturb_hook(perturbation: torch.Tensor):
        def _hook(module: nn.Module, inp: Any, output: Any):
            hidden = _hidden_from_output(output)
            perturbed = hidden + perturbation.to(hidden.dtype)
            if isinstance(output, tuple):
                return (perturbed,) + tuple(output[1:])
            return perturbed
        return _hook

    for module, perturbation in perturbations.items():
        pert_handles.append(module.register_forward_hook(
            _make_perturb_hook(perturbation)))

    try:
        perturbed_loss = task_loss_fn(model, batch)
    finally:
        for h in pert_handles:
            h.remove()

    if cfg.alpha >= 1.0:
        return perturbed_loss
    clean_for_blend = task_loss_fn(model, batch)
    return cfg.alpha * perturbed_loss + (1.0 - cfg.alpha) * clean_for_blend


__all__ = ["TVaccineConfig", "tvaccine_loss"]