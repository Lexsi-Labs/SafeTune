"""
T-Vaccine (Liu et al., 2024, arXiv:2410.09760): a layer-selective variant of Vaccine.

**This is a simplification of the paper's method, not the method.** T-Vaccine (Alg. 1; official
code https://github.com/Lslland/T-Vaccine, ``trainer.py``) computes per-layer gradient norms on a
*harmful* batch every K steps, turns them into sampling probabilities, and every J steps
*randomly samples* gamma layers with those probabilities; only the sampled layers are unfrozen
(LoRA) and perturbed, which is where the memory saving comes from. The T-Vaccine-5 / -8
variants of the paper sample 5 / 8 layers.

What ``tvaccine_loss`` does instead, at every call:

1. forward + backward on the alignment loss; collect the per-layer gradient norms of the
   attention-output hidden states (of the alignment batch itself, not of harmful data);
2. keep the top ``top_k_ratio`` layers by that norm (default 0.5 -- a SafeTune choice, not a
   paper default), deterministically;
3. apply the Vaccine perturbation ``e_r = grad * rho / global_norm_of_selected_grads`` to those
   layers only (this scaling matches the official ``after_first_step``);
4. second forward with the perturbation hooks on the selected layers; minimise that loss.

No layer is frozen, so there is no memory saving. It reuses the Vaccine primitives
(``_get_attention_modules``, ``_hidden_from_output``).
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
            gradient norm. 0.5 = top 50% (a SafeTune default, not the paper's). Range (0, 1].
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

    Simplified relative to Liu et al. 2024 (see the module docstring): only the top-k attention
    layers (by gradient norm of the alignment batch) receive the SAM-style hidden-state
    perturbation, deterministically, instead of probability-sampled layers.
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