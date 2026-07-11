"""
Booster (Huang et al., ICLR 2025 Oral, arXiv:2409.01586).

Official code: https://github.com/git-disl/Booster
(see ``trainer.py`` -> ``BoosterAlignmentTrainer.training_step``).

Booster is an *alignment-stage* defense against harmful fine-tuning. The paper
identifies *harmful perturbation* over the model weights as the root cause of
alignment breaking, and attenuates it by appending a **loss regularizer** to
the alignment objective. The regularizer measures, via a finite difference,
how much the harmful loss *drops* after a simulated one-step harmful attack:

    1. Harmful gradient:    g_h        = grad of harmful loss at w.
    2. Simulated harmful perturbation (single GLOBAL normalization):
                            w'         = w - alpha * g_h / ||g_h||
       where ||g_h|| is the global L2 norm over *all* parameters.
    3. Perturbed harmful gradient:
                            g_h_pert   = grad of harmful loss at w'.
       Then restore the weights: w <- w' + alpha * g_h / ||g_h||.
    4. Alignment gradient:  g_align    = grad of the alignment / SFT loss at w.
    5. Final update gradient:
                            g          = g_align + lambda * (g_h - g_h_pert)

The finite-difference term ``g_h - g_h_pert`` penalises directions along which
a harmful attacker would quickly reduce the harmful loss, making the aligned
model robust to subsequent benign/harmful fine-tuning.

Note: this is **not** a gradient projection. Booster *adds* a scaled harmful
finite-difference term to the alignment gradient; it does not orthogonalize
the alignment gradient against the harmful direction (that is PCGrad/SafeGrad).

Reported: substantial harmful-score reduction vs SFT while preserving
downstream-task accuracy.

Here we expose the regularizer / final-gradient operator only. To compose with
an HF Trainer, perform the harmful backward + perturbation + perturbed backward
inside ``training_step`` and call :func:`booster_project` to combine.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BoosterConfig:
    """Configuration for the Booster alignment-stage regularizer.

    Attributes:
        alpha: magnitude of the simulated harmful perturbation
            ``w' = w - alpha * g_h / ||g_h||`` (paper hyperparameter alpha).
        lamb: weight of the harmful finite-difference regularizer
            ``lambda * (g_h - g_h^perturbed)`` (paper hyperparameter lambda).
        meta_term: if ``True`` use the full finite-difference term
            ``lambda * (g_h - g_h^perturbed)``; if ``False`` use the simplified
            ``lambda * g_h`` variant (the repo's ``meta_term=="False"`` path).
        param_filter: optional list of substrings; only matching parameters
            are regularized. ``None`` regularizes all trainable parameters.
        eps: numerical floor added to the global gradient norm.
        clamp_neg: **deprecated** and ignored. It belonged to the pre-fix
            (incorrect) PCGrad-style projection variant; Booster is an additive
            finite-difference regularizer with no clamping. Accepted only for
            backward compatibility.
    """

    alpha: float = 0.1
    lamb: float = 5.0
    meta_term: bool = True
    param_filter: Optional[list] = None
    eps: float = 1e-7
    clamp_neg: Optional[bool] = None  # deprecated, ignored (pre-fix projection field)


def _global_grad_norm(
    grads: Dict[str, torch.Tensor],
    eps: float = 1e-7,
) -> torch.Tensor:
    """Global L2 norm over a dict of gradient tensors.

    Mirrors Booster's ``_grad_norm``: stack per-tensor L2 norms, then take
    their L2 norm, giving the norm of the concatenated gradient vector.
    """
    if not grads:
        return torch.tensor(eps)
    norms = torch.stack([g.detach().norm(p=2) for g in grads.values()])
    return torch.norm(norms, p=2) + eps


def booster_project(
    grads: Dict[str, torch.Tensor],
    harmful_grads: Dict[str, torch.Tensor],
    perturbed_harmful_grads: Optional[Dict[str, torch.Tensor]] = None,
    config: Optional[BoosterConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Combine the alignment gradient with Booster's harmful regularizer.

    Implements the final-gradient combination of Booster's
    ``BoosterAlignmentTrainer.training_step``::

        g = g_align + lambda * (g_h - g_h^perturbed)

    or, when ``config.meta_term`` is ``False`` (the repo's simplified path)::

        g = g_align + lambda * g_h

    The simulated harmful perturbation that *produces* ``perturbed_harmful_grads``
    is performed by the caller via :func:`booster_perturb_weights` /
    :func:`booster_restore_weights`; this function only combines the resulting
    gradient dicts.

    Args:
        grads: dict ``{name: tensor}`` of the alignment / SFT gradient
            ``g_align`` (the current step's gradient on the alignment loss).
        harmful_grads: dict of the harmful gradient ``g_h`` computed at the
            *unperturbed* weights (see :func:`collect_harmful_gradient`).
        perturbed_harmful_grads: dict of the harmful gradient
            ``g_h^perturbed`` computed *after* the simulated perturbation
            ``w' = w - alpha * g_h / ||g_h||``. If ``None``, the simplified
            ``lambda * g_h`` variant is used regardless of ``config.meta_term``.
        config: :class:`BoosterConfig`.

    Returns:
        New dict ``{name: tensor}`` of the combined Booster update gradient.
    """
    # Backward compatibility: the pre-fix signature was
    # ``booster_project(grads, harmful_grads, config)``. If a caller passes a
    # BoosterConfig in the (new) perturbed_harmful_grads slot, treat it as the
    # config and leave perturbed_harmful_grads unset.
    if isinstance(perturbed_harmful_grads, BoosterConfig):
        config = perturbed_harmful_grads
        perturbed_harmful_grads = None

    cfg = config or BoosterConfig()
    out: Dict[str, torch.Tensor] = {}
    for name, g in grads.items():
        if cfg.param_filter and not any(s in name for s in cfg.param_filter):
            out[name] = g
            continue
        gh = harmful_grads.get(name)
        if gh is None or gh.shape != g.shape:
            out[name] = g
            continue
        # g_align + lambda * g_h
        combined = g.float() + cfg.lamb * gh.float().to(g.device)
        # ... - lambda * g_h^perturbed   (the finite-difference term)
        if cfg.meta_term and perturbed_harmful_grads is not None:
            ghp = perturbed_harmful_grads.get(name)
            if ghp is not None and ghp.shape == g.shape:
                combined = combined - cfg.lamb * ghp.float().to(g.device)
        out[name] = combined.to(g.dtype)
    return out


def booster_perturb_weights(
    model: nn.Module,
    harmful_grads: Dict[str, torch.Tensor],
    config: Optional[BoosterConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Apply Booster's simulated harmful perturbation ``w -= alpha*g_h/||g_h||``.

    Uses a single *global* gradient norm (matching the paper / repo). Returns
    the per-parameter perturbation deltas so the caller can undo them with
    :func:`booster_restore_weights` after computing ``g_h^perturbed``.

    Args:
        model: the model being aligned (modified in-place under ``no_grad``).
        harmful_grads: the harmful gradient ``g_h``.
        config: :class:`BoosterConfig` (uses ``alpha`` and ``eps``).

    Returns:
        Dict ``{name: delta}`` of the applied perturbation (``alpha*g_h/||g_h||``).
    """
    cfg = config or BoosterConfig()
    grad_norm = _global_grad_norm(harmful_grads, cfg.eps)
    deltas: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if cfg.param_filter and not any(s in name for s in cfg.param_filter):
                continue
            gh = harmful_grads.get(name)
            if gh is None or gh.shape != p.shape:
                continue
            delta = (cfg.alpha * gh.to(p.dtype).to(p.device)
                     / grad_norm.to(p.dtype).to(p.device))
            p.data -= delta
            deltas[name] = delta
    return deltas


def booster_restore_weights(
    model: nn.Module,
    deltas: Dict[str, torch.Tensor],
) -> None:
    """Undo a :func:`booster_perturb_weights` perturbation (``w += delta``)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            delta = deltas.get(name)
            if delta is not None and delta.shape == p.shape:
                p.data += delta.to(p.dtype).to(p.device)


@contextmanager
def booster_simulated_perturbation(
    model: nn.Module,
    harmful_grads: Dict[str, torch.Tensor],
    config: Optional[BoosterConfig] = None,
):
    """Context manager: perturb weights along the harmful direction, then restore.

    Inside the ``with`` block the model weights are at ``w' = w - alpha*g_h/||g_h||``;
    recompute the harmful loss/gradient there to obtain ``g_h^perturbed``. On exit
    the original weights are restored.
    """
    deltas = booster_perturb_weights(model, harmful_grads, config)
    try:
        yield
    finally:
        booster_restore_weights(model, deltas)


def collect_harmful_gradient(
    model: nn.Module,
    harmful_batches: Iterable[Dict[str, torch.Tensor]],
    task_loss_fn,
    param_filter: Optional[list] = None,
) -> Dict[str, torch.Tensor]:
    """Compute the averaged harmful-direction gradient ``g_h`` on a small dataset.

    The returned dict is the harmful gradient referenced by
    :func:`booster_project` and :func:`booster_perturb_weights`. In the paper
    this is the gradient of the loss on the (small) harmful pre-set.
    """
    model.zero_grad(set_to_none=True)
    n_batches = 0
    accumulators: Dict[str, torch.Tensor] = {}
    for batch in harmful_batches:
        loss = task_loss_fn(model, batch)
        loss.backward()
        n_batches += 1
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if param_filter and not any(s in name for s in param_filter):
                continue
            accumulators.setdefault(name, torch.zeros_like(p.grad))
            accumulators[name] += p.grad.detach().clone()
        model.zero_grad(set_to_none=True)
    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in accumulators.items()}


__all__ = [
    "BoosterConfig",
    "booster_project",
    "booster_perturb_weights",
    "booster_restore_weights",
    "booster_simulated_perturbation",
    "collect_harmful_gradient",
]
