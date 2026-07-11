"""
CircuitKIT integration for circuit-level observability and targeting.

SafeTune consumes CircuitKIT outputs (via adapter) to enable
circuit-guided SafetyLoRA and neuron/feature-style safety interventions.

The schema is round-trippable: ``load_circuit_info_from_file`` reads a
JSON/YAML file into a :class:`CircuitInfo`, and ``save_circuit_info_to_file``
(or ``CircuitInfo.save``) writes it back to a byte-identical file.
"""

import sys

from .interface import (
    CircuitInfo,
    CircuitInfoProvider,
    LayerModuleSuggestions,
    SafetyRelevantUnits,
)
from .adapter import (
    circuit_info_to_dict,
    get_circuit_info,
    load_circuit_info_from_file,
    save_circuit_info_to_file,
)

__all__ = [
    "CircuitInfo",
    "CircuitInfoProvider",
    "LayerModuleSuggestions",
    "SafetyRelevantUnits",
    "get_circuit_info",
    "load_circuit_info_from_file",
    "circuit_info_to_dict",
    "save_circuit_info_to_file",
]

# ── Top-level re-export ──────────────────────────────────────────────────────
# ``CircuitInfo`` is part of the verified C-ΔΘ public surface and must be
# importable as ``from safetune import CircuitInfo``. This package is imported
# during ``safetune.core``'s import, while the partially-initialised ``safetune``
# module is already in ``sys.modules``; we additively attach the symbol there.
# This keeps the edit confined to ``core/circuit_kit/`` (no parent ``__init__``
# change) and does not alter the ``CircuitInfo`` field signature.
_root = sys.modules.get("safetune")
if _root is not None:
    if not hasattr(_root, "CircuitInfo"):
        _root.CircuitInfo = CircuitInfo  # type: ignore[attr-defined]
    _all = getattr(_root, "__all__", None)
    if isinstance(_all, list) and "CircuitInfo" not in _all:
        _all.append("CircuitInfo")
