"""
Abstract interface for circuit-level information (CircuitKIT integration).

Defines dataclasses/protocols for safety-relevant units, activation patterns,
and layer/module suggestions for LoRA targeting.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class SafetyRelevantUnits:
    """Safety-relevant internal units or features from circuit analysis."""
    layer_indices: List[int] = field(default_factory=list)
    module_names: List[str] = field(default_factory=list)
    unit_ids: List[str] = field(default_factory=list)  # e.g. "L12.MLP.0"
    activation_correlation: Optional[Dict[str, float]] = None  # unit -> correlation with safety
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerModuleSuggestions:
    """Suggested layers/modules for LoRA targeting from CircuitKIT."""
    target_modules: List[str] = field(default_factory=list)  # e.g. ["q_proj", "v_proj"]
    layer_subset: Optional[List[int]] = None  # if not None, only these layers
    priority: Optional[Dict[str, float]] = None  # module -> priority weight
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitInfo:
    """Circuit-level information produced by CircuitKIT (consumed by SafeTune)."""
    safety_units: Optional[SafetyRelevantUnits] = None
    layer_suggestions: Optional[LayerModuleSuggestions] = None
    raw_output_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Writer convenience (additive; field signature unchanged) ─────────────
    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the JSON/YAML dict shape the reader consumes.

        Inverse of ``adapter._dict_to_circuit_info``. ``raw_output_path`` is
        intentionally omitted (the reader derives it from the file path).
        """
        from .adapter import circuit_info_to_dict

        return circuit_info_to_dict(self)

    def save(self, filepath: str, *, fmt: Optional[str] = None, indent: int = 2) -> str:
        """Write this circuit info to a JSON/YAML file (CircuitKIT format).

        Round-trips with ``load_circuit_info_from_file``. See
        ``adapter.save_circuit_info_to_file`` for argument semantics.
        """
        from .adapter import save_circuit_info_to_file

        return save_circuit_info_to_file(self, filepath, fmt=fmt, indent=indent)


class CircuitInfoProvider(Protocol):
    """Protocol for providing circuit info (file, API, or mock)."""

    def get_circuit_info(self, model_name: str, **kwargs: Any) -> Optional[CircuitInfo]:
        """Return circuit info for the given model, or None if unavailable."""
        ...
