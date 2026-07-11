"""
Vaccine (Huang et al., NeurIPS 2024, arXiv:2402.01109).

Idea: while *aligning* the model, Vaccine asks the model to produce hidden
states that are invariant to a small adversarial perturbation. The defining
mechanism (see ``git-disl/Vaccine``, ``trainer.py``) is **not** an input-embedding
perturbation: Vaccine perturbs the **output hidden states of every transformer
attention layer**.

Faithful algorithm, per the authors' ``VaccineTrainer.training_step``:

1. *First pass* — register backward hooks on every attention module. Run the
   alignment loss, backprop, and collect ``grad_output[0]`` for each layer:
   the gradient of the loss w.r.t. that layer's output hidden states.
2. *Perturbation* — take a **single global L2 norm** over all the collected
   per-layer gradients and form, per layer,
   ``e_r = grad * rho / (global_grad_norm + 1e-7)``  (an L2-normalised SAM step).
3. *Second pass* — register forward hooks that add ``e_r`` to each layer's
   output hidden states, then recompute the loss at this worst-case point.
4. The loss to ``.backward()`` is the **perturbed loss alone** — a pure
   SAM-style min-max. There is no separate clean-loss term.

We expose:

* :class:`VaccineConfig`: knobs (perturbation budget ``rho``, ``alpha``).
* :func:`vaccine_loss`: a stateless loss helper. Takes a model, a batch, and a
  reference forward function; returns the Vaccine loss ready to ``.backward()``.

Notes on this stateless factoring vs. the authors' Trainer:

* The authors run *two* full backward passes (the first only to obtain the
  hidden-state gradients, then ``model.zero_grad()``). Here the first pass is
  done with :func:`torch.autograd.grad` so it does not pollute parameter
  ``.grad`` buffers; only the second (perturbed) loss is returned for the
  caller to ``.backward()``. The resulting parameter gradient is identical to
  the authors'.
* ``alpha`` defaults to ``1.0``, which reproduces the canonical paper objective
  (perturbed loss only). Set ``alpha < 1.0`` to blend in a clean-loss term as a
  softer regulariser; this is an opt-in extension, not the paper.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class VaccineConfig:
    """Configuration for the Vaccine loss.

    Attributes:
        rho: L2 budget on the perturbation of the per-layer hidden states.
            The authors' default is ``rho = 2.0`` (``intensity`` in their
            scripts); the perturbation is L2-normalised, so ``rho`` is the
            total L2 step length across all layers.
        alpha: blend weight on the clean-loss term. ``1.0`` reproduces the
            canonical Vaccine objective (perturbed loss only, ``loss = L_pert``).
            ``alpha < 1.0`` returns ``alpha * L_pert + (1 - alpha) * L_clean``
            as a softer regulariser (an opt-in extension, not the paper).
        attn_module_names: substrings used to identify the transformer
            attention modules whose output hidden states are perturbed.
            ``None`` uses the default set covering Llama/OPT/Mistral/GPT-style
            attention class names.
        layer_filter: optional list of name substrings; if given, only
            attention modules whose qualified name contains one of them are
            perturbed. ``None`` = all attention modules (the paper).
        inner_steps: **deprecated** and ignored. Vaccine is a one-step SAM
            ascent (K=1); the pre-fix multi-step field is accepted only for
            backward compatibility and has no effect.
    """

    rho: float = 2.0
    alpha: float = 1.0
    attn_module_names: Optional[List[str]] = None
    layer_filter: Optional[List[str]] = None
    inner_steps: Optional[int] = None  # deprecated, ignored (Vaccine is one-step)


# Class-name substrings for transformer attention modules. Vaccine's
# ``get_leaf_modules_with_grad`` keys on ``LlamaAttention``/``OPTAttention``;
# we widen this to the common HF attention class names so the helper works
# across architectures.
_DEFAULT_ATTN_NAMES = ("Attention", "Attn")


def _get_attention_modules(
    model: nn.Module,
    attn_module_names: Optional[List[str]],
    layer_filter: Optional[List[str]],
) -> List[nn.Module]:
    """Locate the transformer attention modules to perturb.

    Mirrors the authors' ``get_leaf_modules_with_grad``: scan ``named_modules``
    and keep those whose class name marks them as an attention block.
    """
    names = tuple(attn_module_names) if attn_module_names else _DEFAULT_ATTN_NAMES
    modules: List[nn.Module] = []
    for qual_name, module in model.named_modules():
        cls_name = type(module).__name__
        if not any(token in cls_name for token in names):
            continue
        if layer_filter is not None and not any(f in qual_name for f in layer_filter):
            continue
        modules.append(module)
    return modules


def _hidden_from_output(output: Any) -> torch.Tensor:
    """Extract the hidden-state tensor from an attention module's output.

    HF attention modules return either a bare tensor or a tuple whose first
    element is the hidden state (the authors index ``output[0]``).
    """
    if isinstance(output, tuple):
        return output[0]
    return output


def vaccine_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    task_loss_fn: Callable[[Any, Dict[str, torch.Tensor]], torch.Tensor],
    config: Optional[VaccineConfig] = None,
) -> torch.Tensor:
    """Compute the Vaccine perturbation-aware alignment loss.

    ``task_loss_fn(model, batch)`` runs the (alignment-stage) loss. The returned
    scalar is the loss to backprop; only the *perturbed* pass contributes to
    parameter gradients when ``alpha == 1.0`` (the paper's objective).

    Faithful to ``git-disl/Vaccine``: the perturbation is applied to the output
    hidden states of *every* transformer attention layer, scaled by a single
    global L2 norm.
    """
    cfg = config or VaccineConfig()

    attn_modules = _get_attention_modules(model, cfg.attn_module_names, cfg.layer_filter)
    if not attn_modules:
        raise RuntimeError(
            "vaccine_loss: could not locate any transformer attention modules to "
            "perturb. Pass VaccineConfig.attn_module_names with the attention "
            "class-name substring(s) for this architecture."
        )

    # ------------------------------------------------------------------
    # Pass 1: collect the gradient of the loss w.r.t. each attention layer's
    # output hidden states (the authors' backward hooks on grad_output[0]).
    # ------------------------------------------------------------------
    captured_outputs: Dict[nn.Module, torch.Tensor] = {}
    fwd_handles = []

    def _capture_hook(module: nn.Module, inp: Any, output: Any):
        hidden = _hidden_from_output(output)
        # Keep this hidden state in the graph and request its gradient.
        hidden.retain_grad()
        captured_outputs[module] = hidden
        return output

    for module in attn_modules:
        fwd_handles.append(module.register_forward_hook(_capture_hook))

    try:
        clean_loss = task_loss_fn(model, batch)
        ordered = [m for m in attn_modules if m in captured_outputs]
        hiddens = [captured_outputs[m] for m in ordered]
        # grad of the loss w.r.t. each layer's output hidden states.
        grads = torch.autograd.grad(
            clean_loss, hiddens, retain_graph=False, allow_unused=True
        )
    finally:
        for h in fwd_handles:
            h.remove()

    # ------------------------------------------------------------------
    # Build the perturbation: e_r = grad * rho / (global_grad_norm + 1e-7).
    # The norm is a SINGLE L2 norm over the stacked per-layer gradient norms,
    # exactly the authors' `_grad_norm`.
    # ------------------------------------------------------------------
    per_layer_grads: Dict[nn.Module, torch.Tensor] = {}
    for module, grad in zip(ordered, grads):
        if grad is not None:
            per_layer_grads[module] = grad.detach()

    if per_layer_grads:
        global_norm = torch.norm(
            torch.stack([g.norm(p=2) for g in per_layer_grads.values()]), p=2
        )
        scale = cfg.rho / (global_norm + 1e-7)
        perturbations = {m: g * scale for m, g in per_layer_grads.items()}
    else:  # pragma: no cover - no attention layer fed gradient
        logger.warning("vaccine_loss: no hidden-state gradients captured; "
                        "perturbation is empty, falling back to clean loss.")
        perturbations = {}

    # ------------------------------------------------------------------
    # Pass 2: re-run the loss with forward hooks that add e_r to each
    # attention layer's output hidden states. This loss carries the
    # parameter gradient for the alignment update.
    # ------------------------------------------------------------------
    pert_handles = []

    def _make_perturb_hook(perturbation: torch.Tensor):
        def _perturb_hook(module: nn.Module, inp: Any, output: Any):
            hidden = _hidden_from_output(output)
            perturbed = hidden + perturbation.to(hidden.dtype)
            if isinstance(output, tuple):
                return (perturbed,) + tuple(output[1:])
            return perturbed
        return _perturb_hook

    for module, perturbation in perturbations.items():
        pert_handles.append(module.register_forward_hook(_make_perturb_hook(perturbation)))

    try:
        perturbed_loss = task_loss_fn(model, batch)
    finally:
        for h in pert_handles:
            h.remove()

    # Canonical Vaccine (alpha == 1.0): minimise the loss at the worst-case
    # perturbation only. alpha < 1.0 blends in a clean term (opt-in extension).
    if cfg.alpha >= 1.0:
        return perturbed_loss
    clean_for_blend = task_loss_fn(model, batch)
    return cfg.alpha * perturbed_loss + (1.0 - cfg.alpha) * clean_for_blend


__all__ = ["VaccineConfig", "vaccine_loss"]
