"""
Interpret pillar: mechanistic-interpretability primitives for safety.

Closes the Interpret pillar gap in the original plan. Two entry points:

* :func:`identify_safety_neurons`: rank per-layer hidden units by how
  strongly they project onto an external "safety direction" (typically
  the refusal direction from :func:`safetune.steer.extract_refusal_direction`).
  Returns a :class:`safetune.core.circuit_kit.CircuitInfo`, so it slots
  directly into the existing CircuitKit adapter chain.

* :func:`safety_circuit_info`: convenience wrapper that runs refusal-direction
  extraction + safety-neuron localization in one call; the result is read by
  C-ΔΘ and SafetyLoRA.

Both neuron-scoring modes are SafeTune heuristics (badge Variant), not the metric of Chen et al.,
"Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons"
(arXiv:2406.14144, NeurIPS 2025), which compares two checkpoints; see
:mod:`safetune.instrumentation.interpret.safety_neurons` for what differs. The weight mode (cosine of
a ``down_proj`` column with the refusal direction) has no source in that paper.
:func:`eap_safety_circuit` is node-level EAP / EAP-IG (badge Simplified).
"""
from .safety_neurons import (
    SafetyNeuronConfig,
    SafetyNeuronReport,
    identify_safety_neurons,
    safety_circuit_info,
)
from .eap import EAPSafetyCircuitConfig, eap_safety_circuit

__all__ = [
    "SafetyNeuronConfig",
    "SafetyNeuronReport",
    "identify_safety_neurons",
    "safety_circuit_info",
    "EAPSafetyCircuitConfig",
    "eap_safety_circuit",
]
