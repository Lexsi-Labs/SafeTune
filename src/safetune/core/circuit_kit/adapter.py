"""
Adapter that consumes CircuitKIT outputs and returns CircuitInfo.

CircuitKIT is optional; when not installed or when output is missing,
the adapter returns None and SafeTune works without circuits.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .interface import (
    CircuitInfo,
    LayerModuleSuggestions,
    SafetyRelevantUnits,
)

logger = logging.getLogger(__name__)


def load_circuit_info_from_file(filepath: str) -> Optional[CircuitInfo]:
    """
    Load circuit info from a JSON or YAML file (CircuitKIT output format).

    Expected structure (flexible):
    - "safety_units": { "layer_indices": [], "module_names": [], ... }
    - "layer_suggestions": { "target_modules": [], "layer_subset": [] }
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"Circuit file not found: {filepath}")
        return None

    try:
        raw = path.read_text()
        if path.suffix.lower() in (".json",):
            data = json.loads(raw)
        else:
            try:
                import yaml
                data = yaml.safe_load(raw)
            except ImportError:
                data = json.loads(raw)
    except Exception as e:
        logger.warning(f"Failed to parse circuit file {filepath}: {e}")
        return None

    return _dict_to_circuit_info(data, raw_output_path=filepath)


def _dict_to_circuit_info(data: Dict[str, Any], raw_output_path: Optional[str] = None) -> CircuitInfo:
    """Convert dict (e.g. from JSON/YAML) to CircuitInfo."""
    safety_units = None
    su = data.get("safety_units") or data.get("safety_relevant_units")
    if su and isinstance(su, dict):
        safety_units = SafetyRelevantUnits(
            layer_indices=su.get("layer_indices", []),
            module_names=su.get("module_names", []),
            unit_ids=su.get("unit_ids", []),
            activation_correlation=su.get("activation_correlation"),
            metadata=su.get("metadata", {}),
        )

    layer_suggestions = None
    ls = data.get("layer_suggestions") or data.get("lora_suggestions")
    if ls and isinstance(ls, dict):
        layer_suggestions = LayerModuleSuggestions(
            target_modules=ls.get("target_modules", []),
            layer_subset=ls.get("layer_subset"),
            priority=ls.get("priority"),
            metadata=ls.get("metadata", {}),
        )

    return CircuitInfo(
        safety_units=safety_units,
        layer_suggestions=layer_suggestions,
        raw_output_path=raw_output_path,
        metadata=data.get("metadata", {}),
    )


def get_circuit_info(
    source: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> Optional[CircuitInfo]:
    """
    Get circuit info from file path or (future) API.

    Args:
        source: Path to CircuitKIT output file, or None to return None.
        model_name: Optional model name for API lookup (not used for file).
        **kwargs: Passed to future API adapter.

    Returns:
        CircuitInfo or None if unavailable.
    """
    if not source:
        return None
    if Path(source).exists() or source.endswith(".json") or source.endswith(".yaml"):
        return load_circuit_info_from_file(source)
    return None


# ── Writer ──────────────────────────────────────────────────────────────────
# Inverse of the reader above. ``circuit_info_to_dict`` produces exactly the
# dict shape that ``_dict_to_circuit_info`` consumes, so
#   load_circuit_info_from_file -> save_circuit_info_to_file
# round-trips to an identical file. ``raw_output_path`` is intentionally NOT
# serialised: the reader derives it from the file path it was given, so writing
# it would break the round-trip. Nested ``metadata`` dicts are emitted only
# when non-empty, matching the reader's ``.get(..., {})`` defaults.


def circuit_info_to_dict(info: CircuitInfo) -> Dict[str, Any]:
    """
    Serialise a :class:`CircuitInfo` to a plain dict.

    The result is the canonical (preferred-spelling) form accepted by
    :func:`_dict_to_circuit_info` / :func:`load_circuit_info_from_file`.
    """
    data: Dict[str, Any] = {}

    su = info.safety_units
    if su is not None:
        su_dict: Dict[str, Any] = {
            "layer_indices": list(su.layer_indices),
            "module_names": list(su.module_names),
            "unit_ids": list(su.unit_ids),
        }
        if su.activation_correlation is not None:
            su_dict["activation_correlation"] = dict(su.activation_correlation)
        if su.metadata:
            su_dict["metadata"] = dict(su.metadata)
        data["safety_units"] = su_dict

    ls = info.layer_suggestions
    if ls is not None:
        ls_dict: Dict[str, Any] = {
            "target_modules": list(ls.target_modules),
        }
        if ls.layer_subset is not None:
            ls_dict["layer_subset"] = list(ls.layer_subset)
        if ls.priority is not None:
            ls_dict["priority"] = dict(ls.priority)
        if ls.metadata:
            ls_dict["metadata"] = dict(ls.metadata)
        data["layer_suggestions"] = ls_dict

    if info.metadata:
        data["metadata"] = dict(info.metadata)

    return data


def save_circuit_info_to_file(
    info: CircuitInfo,
    filepath: str,
    *,
    fmt: Optional[str] = None,
    indent: int = 2,
) -> str:
    """
    Write a :class:`CircuitInfo` to a JSON or YAML file (CircuitKIT format).

    Args:
        info: The circuit info to serialise.
        filepath: Destination path. The format is inferred from the suffix
            (``.json`` -> JSON, ``.yaml`` / ``.yml`` -> YAML, else JSON)
            unless ``fmt`` is given.
        fmt: Explicit format override: ``"json"`` or ``"yaml"``.
        indent: JSON indentation (ignored for YAML).

    Returns:
        The path written, as a string.

    The output is exactly what :func:`load_circuit_info_from_file` consumes, so
    ``load -> save`` produces a byte-for-byte identical file (modulo a single
    trailing newline appended for POSIX-friendliness).
    """
    path = Path(filepath)
    data = circuit_info_to_dict(info)

    if fmt is None:
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            fmt = "yaml"
        else:
            fmt = "json"
    fmt = fmt.lower()

    if fmt == "yaml":
        try:
            import yaml
        except ImportError:
            logger.warning(
                "PyYAML not installed; writing JSON to %s instead of YAML", filepath
            )
            text = json.dumps(data, indent=indent, sort_keys=False) + "\n"
        else:
            text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    elif fmt == "json":
        text = json.dumps(data, indent=indent, sort_keys=False) + "\n"
    else:
        raise ValueError(f"Unknown format {fmt!r}; expected 'json' or 'yaml'")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    logger.debug("Wrote circuit info to %s (%s)", filepath, fmt)
    return str(path)
