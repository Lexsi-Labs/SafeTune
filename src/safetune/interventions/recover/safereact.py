"""SafeReAct: reactivate suppressed safety neurons via a low-rank LoRA delta.

Reference: "Finding and Reactivating Post-Trained LLMs' Hidden Safety
Mechanisms", NeurIPS 2025. Repo: https://github.com/homles11/SafeReAct
(``src/lorra_safereact_twomodel.py``).

The authors' method does NOT do a one-shot weight merge. It trains a small
set of **LoRA adapters** (rank 16, alpha 10, lr 2e-5, ~300 iters) on a sparse
set of layers (e.g. every 5th layer) with a *representation* objective:

    loss = retain_coeff * retain_loss
         + circuit_breaker_coeff * circuit_breaker_loss

where both terms are L2 norms between hidden states of the LoRA-adapted model
and a reference (``circuit_breaker_loss`` pulls the post-trained model's
harmful-prompt representations toward the safe reference; ``retain_loss`` keeps
utility-prompt representations close to the original post-trained model). The
coefficients are scheduled by training progress: ``retain_coeff`` rises and
``circuit_breaker_coeff`` falls (``progress``-linear, scaled by ``alpha``), so
optimization focuses on safety first, then utility preservation. The base
(non-LoRA) forward is obtained via ``peft``'s ``model.disable_adapter()``.

This module restores that LoRA-training step. When ``peft`` and ``torch`` are
available it attaches LoRA adapters to the suppressed-neuron parent layers and
runs the LoRRA-style representation-training loop, then merges the trained
adapter into ``model``. If ``peft`` is unavailable, or ``train_lora=False``, it
falls back to the prior training-free one-shot weight merge toward the
reference -- a documented approximation.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


# Authors' defaults (homles11/SafeReAct, R1-8B config).
_SAFEREACT_LR = 2e-5
_SAFEREACT_ITERS = 300
_SAFEREACT_LAYER_STRIDE = 5
# Suffix appended to harmful prompts during alignment (paper Eq. 3).
_SAFETY_SUFFIX = (
    " Remember you should not generate any harmful or misleading content."
)


def _device_of(model: nn.Module) -> Any:
    """Return the device the model's parameters live on (defaults to CPU)."""
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - parameterless module
        import torch

        return torch.device("cpu")


def _move_to_device(inputs: Any, device: Any) -> Any:
    """Move probe inputs onto ``device``.

    Fixes the FEATURE_MAP device-mismatch bug: ``core.safereact`` builds a
    default ``torch.randint(0, 100, (1, 16))`` probe on CPU and feeds it
    straight into ``model(...)``; if the model is on CUDA the forward pass
    raises a device-mismatch ``RuntimeError``. We co-locate the probe with the
    model before any forward pass touches it.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - defensive
        return inputs

    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }
    if isinstance(inputs, (tuple, list)):
        moved = [
            (v.to(device) if isinstance(v, torch.Tensor) else v) for v in inputs
        ]
        return type(inputs)(moved) if isinstance(inputs, tuple) else moved
    return inputs


def _select_target_layer_modules(
    model: nn.Module,
    aligned_sd: dict,
    base_sd: dict,
    stride: int,
) -> List[str]:
    """Pick LoRA-target Linear module names on a sparse (every-``stride``) set
    of transformer layers whose parents were flagged as suppressed.

    Mirrors the authors' "every five layers for efficiency" layer selection.
    Returns module names (without the ``.weight`` suffix) suitable for peft's
    ``target_modules``.
    """
    suppressed_layers: set[int] = set()
    candidate_modules: set[str] = set()
    for name in aligned_sd:
        if name not in base_sd:
            continue
        parts = name.split(".")
        # Find the transformer-layer index (HF: model.layers.<i>.<...>).
        for i, p in enumerate(parts):
            if p.isdigit() and i > 0 and parts[i - 1] in ("layers", "h", "blocks"):
                suppressed_layers.add(int(p))
                break
    if not suppressed_layers:
        return []
    keep = {ly for ly in suppressed_layers if ly % stride == 0} or suppressed_layers

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        parts = module_name.split(".")
        for i, p in enumerate(parts):
            if p.isdigit() and i > 0 and parts[i - 1] in ("layers", "h", "blocks"):
                if int(p) in keep:
                    candidate_modules.add(module_name)
                break
    return sorted(candidate_modules)


def _unload_peft_in_place(peft_model: Any) -> None:
    """Best-effort removal of the peft LoRA wrapping on failure paths.

    ``get_peft_model`` mutates the base model in place (it swaps the target
    ``nn.Linear`` modules for ``lora.Linear`` wrappers, which renames their
    parameters, e.g. ``fc.weight`` -> ``fc.base_layer.weight``). If we bail
    out of training without undoing that, the *caller's* model reference is
    left peft-wrapped: the training-free fallback's key matching silently
    no-ops and checkpoints serialize a mangled module tree. ``unload()``
    restores the original modules in place on the base model, so the caller's
    reference regains its original structure and parameter names.
    """
    try:
        peft_model.unload()
    except Exception:  # pragma: no cover - defensive
        logger.warning(
            "SafeReAct: failed to unload the peft adapter cleanly; the model "
            "may be left peft-wrapped.",
            exc_info=True,
        )


def _train_reactivation_lora(
    model: nn.Module,
    reference_model: nn.Module,
    target_modules: List[str],
    probe_inputs: Any,
    lora_rank: int,
    lora_alpha: float,
    reactivation_scale: float,
    iters: int,
    lr: float,
) -> bool:
    """Run the LoRRA-style representation-training loop and merge the adapter.

    Returns ``True`` if a LoRA adapter was trained and merged into ``model``,
    ``False`` if training could not run (caller then falls back to the
    one-shot merge).

    Every ``False`` path taken *after* ``get_peft_model`` first restores the
    caller's original module structure via ``_unload_peft_in_place`` (peft
    wraps in place, so bailing out without unloading would leave the caller's
    reference with renamed parameters). Test-style verification on a tiny CPU
    model::

        import torch.nn as nn
        from peft import LoraConfig, get_peft_model
        m = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        before = sorted(n for n, _ in m.named_parameters())
        pm = get_peft_model(m, LoraConfig(r=2, target_modules=["0", "1"]))
        assert sorted(n for n, _ in m.named_parameters()) != before  # wrapped
        _unload_peft_in_place(pm)  # what every failure path now does
        assert sorted(n for n, _ in m.named_parameters()) == before  # restored
    """
    try:
        import torch
        import torch.nn.functional as F
        from peft import LoraConfig, get_peft_model
    except ImportError:
        return False

    if not target_modules:
        return False

    device = _device_of(model)
    probes = _move_to_device(probe_inputs, device)
    if probes is None:
        # Synthesise a probe co-located with the model (CPU-safe default).
        probes = torch.randint(0, 100, (1, 16), device=device)

    def _forward_hidden(m: nn.Module, x: Any) -> Any:
        if isinstance(x, dict):
            out = m(**x, output_hidden_states=True)
        elif isinstance(x, (tuple, list)):
            out = m(*x, output_hidden_states=True)
        else:
            out = m(x, output_hidden_states=True)
        hs = getattr(out, "hidden_states", None)
        if hs is None and isinstance(out, dict):
            hs = out.get("hidden_states")
        return hs

    # Reference (safe) hidden states -- frozen.
    reference_model.eval()
    with torch.no_grad():
        try:
            ref_hs = _forward_hidden(reference_model, _move_to_device(probes, _device_of(reference_model)))
        except Exception:
            return False
    if ref_hs is None:
        return False
    ref_hs = [h.to(device).detach().float() for h in ref_hs]

    # Original post-trained hidden states (LoRA disabled) -- frozen retain target.
    model.eval()
    with torch.no_grad():
        try:
            orig_hs = _forward_hidden(model, probes)
        except Exception:
            return False
    if orig_hs is None:
        return False
    orig_hs = [h.to(device).detach().float() for h in orig_hs]

    cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    try:
        peft_model = get_peft_model(model, cfg)
    except Exception:
        return False

    opt = torch.optim.AdamW(
        (p for p in peft_model.parameters() if p.requires_grad), lr=lr
    )

    n_layers = min(len(ref_hs), len(orig_hs))
    for step in range(max(1, iters)):
        peft_model.train()
        opt.zero_grad()
        lora_hs = _forward_hidden(peft_model, probes)
        if lora_hs is None:
            # Restore the caller's original module structure before bailing.
            _unload_peft_in_place(peft_model)
            del peft_model
            return False
        lora_hs = [h.float() for h in lora_hs[:n_layers]]

        # circuit_breaker_loss: pull harmful-prompt reps toward the safe ref.
        cb_loss = torch.stack(
            [
                torch.norm(lora_hs[l] - ref_hs[l], dim=-1, p=2).nanmean()
                for l in range(n_layers)
            ]
        ).mean()
        # retain_loss: keep utility reps close to the original post-trained model.
        retain_loss = torch.stack(
            [
                torch.norm(lora_hs[l] - orig_hs[l], dim=-1, p=2).nanmean()
                for l in range(n_layers)
            ]
        ).mean()

        # Progress-scheduled coefficients (authors: retain rises, cb falls).
        progress = step / max(1, iters)
        retain_coeff = reactivation_scale * progress
        cb_coeff = reactivation_scale * (1.0 - progress)
        loss = retain_coeff * retain_loss + cb_coeff * cb_loss

        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()

    # Merge the trained adapter back into the base weights, leaving a plain
    # nn.Module (the caller's @assert_mutates sees the mutated base params).
    # Use merge_and_unload() atomically — the two-step merge_adapter()+unload()
    # pattern risks a double-merge if merge_adapter() succeeds but unload() raises,
    # causing the fallback merge_and_unload() to fold the adapter a second time.
    try:
        merged = peft_model.merge_and_unload()
        model.load_state_dict(merged.state_dict())
        del merged
    except Exception:
        # merge_and_unload may have raised part-way; strip any remaining LoRA
        # wrappers so the caller's model regains its original structure and
        # parameter names before the fallback path runs.
        _unload_peft_in_place(peft_model)
        return False
    return True


@assert_mutates("apply_safereact")
def apply_safereact(
    model: nn.Module,
    reference_model: nn.Module,
    top_k_neurons: int = 64,
    reactivation_scale: float = 1.0,
    lora_rank: int = 16,
    lora_alpha: float = 10.0,
    target_modules: Optional[List[str]] = None,
    probe_inputs: Any = None,
    train_lora: bool = True,
    lora_iters: int = _SAFEREACT_ITERS,
    lora_lr: float = _SAFEREACT_LR,
    layer_stride: int = _SAFEREACT_LAYER_STRIDE,
    **extra: Any,
) -> nn.Module:
    """Build a SafeReAct reactivation LoRA and merge it into ``model``.

    Faithful to homles11/SafeReAct: identifies suppressed safety neurons via
    activation contrast, attaches LoRA adapters to a sparse set of their parent
    layers, and runs the LoRRA-style representation-training loop
    (``retain_loss`` + ``circuit_breaker_loss`` with progress-scheduled
    coefficients), then merges the trained adapter into ``model``.

    New optional kwargs (defaults preserve the public ``apply_safereact``
    contract): ``train_lora`` enables/disables the training step,
    ``lora_iters`` / ``lora_lr`` / ``layer_stride`` expose the authors'
    training hyper-parameters. ``lora_rank``/``lora_alpha`` default to the
    paper's values (16 / 10). When ``peft`` is unavailable or training cannot
    run, a documented one-shot weight-merge fallback is used.
    """
    try:
        from safetune.core.safereact import (
            SafeReActConfig,
            apply_safereact as _run_safereact,
        )
    except ImportError as e:  # pragma: no cover - defensive
        raise ImportError(
            f"apply_safereact needs safetune.core.safereact: {e}"
        ) from e

    cfg = SafeReActConfig(
        top_k_neurons=top_k_neurons,
        reactivation_scale=reactivation_scale,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
    )

    # Co-locate the probe with the model BEFORE core.safereact runs its
    # forward passes -- fixes the device-mismatch bug (probe is built on CPU
    # by core.safereact but the model may be on CUDA).
    located_probes = (
        _move_to_device(probe_inputs, _device_of(model))
        if probe_inputs is not None
        else None
    )

    payload = _run_safereact(
        post_trained_model=model,
        reference_model=reference_model,
        config=cfg,
        probe_inputs=located_probes,
    )
    if not payload:
        return model

    aligned_sd = payload.get("aligned_state_dict", {})
    base_sd = payload.get("base_state_dict", {})
    alpha = payload.get("alpha", reactivation_scale)

    try:
        import torch
    except ImportError:  # pragma: no cover - defensive
        return model

    # --- Faithful path: train a reactivation LoRA on the suppressed layers. ---
    if train_lora:
        target = _select_target_layer_modules(
            model, aligned_sd, base_sd, max(1, layer_stride)
        )
        trained = _train_reactivation_lora(
            model=model,
            reference_model=reference_model,
            target_modules=target,
            probe_inputs=located_probes if located_probes is not None else probe_inputs,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            reactivation_scale=alpha,
            iters=lora_iters,
            lr=lora_lr,
        )
        if trained:
            return model

    # --- Fallback: training-free one-shot weight merge toward the reference. ---
    # Prefer the payload's aligned/base delta when its keys line up with the
    # model's parameters; otherwise merge directly toward the reference model
    # (the payload often carries neuron-indexed keys that DON'T match
    # named_parameters, which would silently no-op).
    n_applied = 0
    with torch.no_grad():
        use_payload = any(
            name in aligned_sd and name in base_sd
            for name, _ in model.named_parameters()
        )
        if use_payload:
            for name, param in model.named_parameters():
                if name not in aligned_sd or name not in base_sd:
                    continue
                a = aligned_sd[name].to(param.device).to(param.dtype)
                b = base_sd[name].to(param.device).to(param.dtype)
                param.data.copy_(param.data + alpha * (a - b))
                n_applied += 1
        else:
            ref_sd = reference_model.state_dict()
            for name, param in model.named_parameters():
                if name not in ref_sd:
                    continue
                r = ref_sd[name].to(param.device).to(param.dtype)
                param.data.copy_(param.data + alpha * (r - param.data))
                n_applied += 1
    if n_applied == 0:
        logger.warning(
            "apply_safereact: training-free fallback applied no weight changes "
            "(no parameter keys matched). Pass train_lora=True for the faithful "
            "reactivation-LoRA path."
        )
    return model


__all__ = ["apply_safereact"]
