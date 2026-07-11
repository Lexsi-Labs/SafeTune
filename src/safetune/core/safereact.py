"""
SafeReAct: Finding and Reactivating Post-Trained LLMs' Hidden Safety Mechanisms
(weight-arithmetic variant).

Reference paper: "Finding and Reactivating Post-Trained LLMs' Hidden Safety
Mechanisms" (NeurIPS 2025).  Canonical code at homles11/SafeReAct uses
LoRRA-style representation training (Circuit-Breakers-flavoured) on
two models with hidden-state cosine-similarity losses; that approach
requires a training loop and DeepSpeed.

This module implements a *lighter* variant: it identifies safety neurons
that have been suppressed by post-training using activation contrast on
probe inputs, then merges the reference weights for those neurons back
into the post-trained model.  No training, no representation loss.  Use
this when you need a fast training-free patch; use the LoRRA reference
for full paper fidelity.

Bug-fix history: in earlier versions, ``build_reactivation_lora`` pre-blended
``base_sd`` by the same ``reactivation_scale`` that was then re-applied at
merge time, which made the default ``scale=1.0`` a silent no-op.  The
state-dict mutation diagnostic at ``tests/diagnostic/`` confirms this and
prevents regression.

Integration with SafeTune:
    - Uses ``ActivationCapture`` from ``neuron_ft.py`` to collect activations.
    - ``build_reactivation_lora()`` produces a LoRA state-dict that can be fed
      directly into ``SafeLoRAPatch.apply_to_model()`` for seamless patching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SafeReActConfig:
    """Configuration for SafeReAct hidden-safety-mechanism reactivation.

    Args:
        reference_model_path: Path to the original *aligned* checkpoint
            (used as the "safe" activation reference).
        top_k_neurons: Number of suppressed neurons to target.
        reactivation_scale: Blend factor in [0,1] — 1.0 fully restores
            reference activations; 0.0 is a no-op.
        lora_rank: Rank for the reactivation LoRA adapter.
        lora_alpha: LoRA alpha scaling factor.
        target_modules: Submodule name patterns to search for suppressed
            neurons (``None`` = all named modules).
        probe_prompts: Optional list of prompt strings used to collect
            activations. If ``None``, random tensors are used for testing.
    """
    reference_model_path: str = ""
    top_k_neurons: int = 64
    reactivation_scale: float = 1.0
    lora_rank: int = 8
    lora_alpha: float = 16.0
    target_modules: Optional[List[str]] = None
    probe_prompts: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_safereact_config(**kwargs: Any) -> SafeReActConfig:
    """Build config from dict (e.g. YAML safety.methods.params)."""
    valid = {f.name for f in SafeReActConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return SafeReActConfig(**{k: v for k, v in kwargs.items() if k in valid})


# ---------------------------------------------------------------------------
# Step 1: Contrast activations to find suppressed safety neurons
# ---------------------------------------------------------------------------

def find_suppressed_safety_neurons(
    post_trained_model: Any,
    reference_model: Any,
    config: SafeReActConfig,
    probe_inputs: Optional[Any] = None,
) -> List[Any]:
    """Identify neurons suppressed in ``post_trained_model`` relative to ``reference_model``.

    A neuron is *suppressed* when its mean absolute activation in the
    post-trained model is substantially *lower* than in the reference model.
    The suppression score is ``ref_activation - post_activation`` (clamped ≥ 0).

    Args:
        post_trained_model: The fine-tuned / post-RLHF ``nn.Module``.
        reference_model: The original aligned ``nn.Module`` (loaded from
            ``config.reference_model_path`` if not provided directly).
        config: ``SafeReActConfig``.
        probe_inputs: Tensor or dict passed to ``model.forward()``. If
            ``None``, a random ``(1, 16)`` int tensor is used.

    Returns:
        List of ``NeuronUnitScore`` sorted by suppression magnitude (desc).
    """
    from .neuron_ft import (
        NeuronUnitScore,
        collect_activations,
    )

    try:
        import torch
        probes: Any = probe_inputs
        if probes is None:
            # Co-locate default probes with the model so localization doesn't
            # crash on a GPU model (probe_inputs=None is the first-timer path).
            try:
                _dev = next(post_trained_model.parameters()).device
            except StopIteration:
                _dev = "cpu"
            probes = torch.randint(0, 100, (1, 16), device=_dev)
    except ImportError:
        logger.warning("SafeReAct: torch not available, returning empty neuron list.")
        return []

    # Collect activations from both models
    logger.info("SafeReAct: collecting post-trained model activations…")
    post_acts = collect_activations(
        post_trained_model, probes, module_names=config.target_modules
    )
    logger.info("SafeReAct: collecting reference model activations…")
    ref_acts = collect_activations(
        reference_model, probes, module_names=config.target_modules
    )

    # Score: how much has each neuron been suppressed?
    scores: List[NeuronUnitScore] = []
    for unit_id in set(ref_acts) | set(post_acts):
        ref_val  = ref_acts.get(unit_id, 0.0)
        post_val = post_acts.get(unit_id, 0.0)
        suppression = max(0.0, ref_val - post_val)
        scores.append(NeuronUnitScore(unit_id=unit_id, score=suppression))

    scores.sort(key=lambda x: x.score, reverse=True)
    top_k = scores[: max(0, config.top_k_neurons)]
    logger.info("SafeReAct: identified %d suppressed neuron(s).", len(top_k))
    return top_k


# ---------------------------------------------------------------------------
# Step 2: Build a reactivation LoRA state-dict for the suppressed neurons
# ---------------------------------------------------------------------------

def build_reactivation_lora(
    suppressed_units: List[Any],
    post_trained_model: Any,
    reference_model: Any,
    config: SafeReActConfig,
) -> Dict[str, Any]:
    """Construct a LoRA state-dict that boosts suppressed safety neurons.

    The LoRA adapter is built by computing the low-rank decomposition of the
    *delta* between the reference and post-trained weights for each targeted
    parameter, scaled by ``config.reactivation_scale``.  This is compatible
    with ``SafeLoRAPatch.apply_to_model()`` in *state_dict* mode.

    Args:
        suppressed_units: Output of ``find_suppressed_safety_neurons()``.
        post_trained_model: The post-trained model whose weights will be patched.
        reference_model: The aligned reference model.
        config: ``SafeReActConfig``.

    Returns:
        Dict ``{"aligned_state_dict": {...}, "base_state_dict": {...}}``
        ready for ``SafeLoRAPatch`` with ``alpha=config.reactivation_scale``.
    """
    if not suppressed_units:
        logger.info("SafeReAct: no suppressed units; returning identity LoRA.")
        return {}

    try:
        import torch
    except ImportError:
        logger.warning("SafeReAct: torch not available, cannot build LoRA.")
        return {}

    # Map unit_ids to param names (strip <root> sentinel if present)
    target_ids = {u.unit_id for u in suppressed_units}

    aligned_sd: Dict[str, Any] = {}
    base_sd: Dict[str, Any] = {}

    ref_state  = {n: p.detach().clone() for n, p in reference_model.named_parameters()}
    post_state = {n: p.detach().clone() for n, p in post_trained_model.named_parameters()}

    for name in ref_state:
        # Include param if its parent module is in our target set
        parent = ".".join(name.split(".")[:-1]) or "<root>"
        if parent not in target_ids and name not in target_ids:
            continue
        if name not in post_state:
            continue
        # Store unscaled reference and post values. The blend factor
        # (config.reactivation_scale) is propagated as ``alpha`` and applied
        # downstream by ``recover.safereact.apply_safereact``:
        #
        #     new_w = post_w + alpha * (ref_w - post_w)
        #
        # With alpha = 1.0 this fully restores the reference value for the
        # suppressed neurons; with alpha = 0.0 it is a no-op. The earlier
        # implementation pre-blended ``base_sd`` here AND multiplied by
        # ``alpha`` at apply time, which double-counted the scale and reduced
        # alpha = 1.0 to a zero-delta no-op.
        aligned_sd[name] = ref_state[name]
        base_sd[name]    = post_state[name]

    logger.info(
        "SafeReAct: built reactivation LoRA covering %d parameter(s).", len(aligned_sd)
    )
    return {
        "aligned_state_dict": aligned_sd,
        "base_state_dict": base_sd,
        "alpha": config.reactivation_scale,
        "neuron_unit_ids": [u.unit_id for u in suppressed_units],
    }


# ---------------------------------------------------------------------------
# Convenience: end-to-end SafeReAct pipeline
# ---------------------------------------------------------------------------

def apply_safereact(
    post_trained_model: Any,
    reference_model: Any,
    config: SafeReActConfig,
    probe_inputs: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run full SafeReAct pipeline and return the reactivation LoRA payload.

    This is a convenience wrapper that chains:
    1. ``find_suppressed_safety_neurons()``
    2. ``build_reactivation_lora()``

    The returned dict can be passed to ``SafeLoRAPatch(params=result).apply_to_model(model)``.
    """
    units = find_suppressed_safety_neurons(
        post_trained_model, reference_model, config, probe_inputs=probe_inputs
    )
    return build_reactivation_lora(units, post_trained_model, reference_model, config)
