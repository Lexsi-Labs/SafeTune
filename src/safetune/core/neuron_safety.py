"""
Neuron/feature-style safety: use CircuitInfo to identify safety-relevant units,
validate impact (ablation or activation correlation), and apply targeted interventions.

CircuitKIT produces circuit-level information; this module consumes it for
targeting and optional intervention hooks.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .circuit_kit import CircuitInfo, SafetyRelevantUnits, LayerModuleSuggestions


@dataclass
class NeuronSafetyResult:
    """Result of safety-relevant unit identification or validation."""
    safety_units: Optional[SafetyRelevantUnits] = None
    layer_suggestions: Optional[LayerModuleSuggestions] = None
    validation_score: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def identify_safety_units(circuit_info: Optional[CircuitInfo]) -> Optional[SafetyRelevantUnits]:
    """Extract safety-relevant units from CircuitInfo."""
    if circuit_info is None or circuit_info.safety_units is None:
        return None
    return circuit_info.safety_units


def get_lora_targeting_from_circuit(circuit_info: Optional[CircuitInfo]) -> Optional[LayerModuleSuggestions]:
    """Get LoRA module suggestions from CircuitInfo for SafetyLoRA."""
    if circuit_info is None or circuit_info.layer_suggestions is None:
        return None
    return circuit_info.layer_suggestions


def validate_impact(
    circuit_info: Optional[CircuitInfo],
    safety_score_fn: Optional[Callable[[List[str]], List[float]]] = None,
    ablation_fn: Optional[Callable[..., float]] = None,
) -> float:
    """
    Validate impact of safety-relevant units via ablation, safety scoring, or activation
    correlation.  Returns a validation score in [0, 1], or 0.0 when no data is available.

    Priority:
    1. If ablation_fn is provided, call ablation_fn(circuit_info) and return the result.
    2. Else if safety_score_fn is provided and unit_ids are available, call
       safety_score_fn(unit_ids) and return the mean score.
    3. Else if activation_correlation is non-empty, return the mean absolute correlation.
    4. Otherwise return 0.0.
    """
    if circuit_info is None:
        return 0.0

    # Priority 1: explicit ablation function
    if ablation_fn is not None:
        try:
            return float(ablation_fn(circuit_info))
        except Exception:
            pass

    units = circuit_info.safety_units

    # Priority 2: safety score function over identified unit ids
    if safety_score_fn is not None and units is not None and units.unit_ids:
        try:
            scores = safety_score_fn(units.unit_ids)
            if scores:
                return float(sum(scores) / len(scores))
        except Exception:
            pass

    # Priority 3: mean absolute activation correlation
    if units is not None and units.activation_correlation:
        corrs = list(units.activation_correlation.values())
        if corrs:
            return float(sum(abs(c) for c in corrs) / len(corrs))

    return 0.0


def apply_circuit_to_safety_lora(
    circuit_info: Optional[CircuitInfo],
    safety_lora_config: Any,
) -> Any:
    """
    Wire CircuitKIT output into SafetyLoRA: set target_modules from layer_suggestions.
    safety_lora_config: SafetyLoRAConfig or dict with lora_target_modules.
    Returns updated config (same object if mutable, or new dict).
    """
    if circuit_info is None or circuit_info.layer_suggestions is None:
        return safety_lora_config
    sug = circuit_info.layer_suggestions
    if not sug.target_modules:
        return safety_lora_config
    if hasattr(safety_lora_config, "lora_target_modules"):
        safety_lora_config.lora_target_modules = list(sug.target_modules)
        safety_lora_config.circuit_info = circuit_info
        safety_lora_config.circuit_guided = True
        return safety_lora_config
    if isinstance(safety_lora_config, dict):
        out = dict(safety_lora_config)
        out["lora_target_modules"] = list(sug.target_modules)
        out["circuit_guided"] = True
        out["circuit_info"] = circuit_info
        return out
    return safety_lora_config
