"""
Interpret pillar: mechanistic-interpretability primitives for safety.

Closes the Interpret pillar gap in the original plan. Two entry points:

* :func:`identify_safety_neurons`: rank per-layer hidden units by how
  strongly they project onto an external "safety direction" (typically
  the refusal direction from :func:`safetune.steer.extract_refusal_direction`).
  Returns a :class:`safetune.core.circuit_kit.CircuitInfo`, so it slots
  directly into the existing CircuitKit adapter chain.

* :func:`safety_circuit_info`: convenience wrapper that runs refusal-direction
  extraction + safety-neuron localization in one call, suitable for handing
  directly to LSSF / PKE / DeepRefusal.

The "safety neuron" definition matches the convention of Chen et al.
(NeurIPS 2024) and the Arditi et al. appendix: a neuron is safety-relevant
when its output projection has high cosine with the refusal direction in
that layer.
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
