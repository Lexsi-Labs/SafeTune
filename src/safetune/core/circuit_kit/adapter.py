"""
Adapter that consumes CircuitKIT outputs and returns CircuitInfo.

CircuitKIT is optional; when not installed or when output is missing,
the adapter returns None and SafeTune works without circuits.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from .interface import (
    CircuitInfo,
    LayerModuleSuggestions,
    SafetyRelevantUnits,
)

logger = logging.getLogger(__name__)


def load_circuit_info_from_file(filepath: str, *, top_fraction: float = 0.2) -> Optional[CircuitInfo]:
    """
    Load circuit info from a JSON or YAML file (CircuitKIT output format).

    Expected structure (flexible):
    - "safety_units": { "layer_indices": [], "module_names": [], ... }
    - "layer_suggestions": { "target_modules": [], "layer_subset": [] }
    - or only CircuitKIT's "node_scores" ({node name: score}, e.g. a
      ``*_scores.json`` from CircuitKIT < 0.2): both are then derived from the
      top ``top_fraction`` of nodes by score (see ``circuit_info_from_node_scores``).

    A missing file returns ``None`` (with a warning). Any unreadable or invalid file raises
    ``ValueError`` naming the path and the reason: it cannot be parsed, it is not a mapping (an
    empty file, a list, a scalar), it has an unknown key inside ``safety_units`` /
    ``layer_suggestions``, ``layer_indices`` / ``layer_subset`` are not non-negative integers, or it
    has none of the keys above. An empty mapping ``{}`` loads as an empty ``CircuitInfo``.
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
        raise ValueError(f"Cannot read circuit file {filepath}: {type(e).__name__}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(
            f"Circuit file {filepath} must hold a mapping of circuit keys, got "
            f"{'nothing (empty file)' if data is None else type(data).__name__}."
        )
    return _dict_to_circuit_info(data, raw_output_path=filepath, top_fraction=top_fraction)


# CircuitKIT node names: "A3.5" (layer 3, head 5), "MLP 3" (node) or "L3.42" (neuron).
_HEAD = re.compile(r"A(\d+)\.(\d+)$")
_MLP = re.compile(r"(?:MLP |L)(\d+)(?:\.\d+)?$")
# Hugging Face module names of Llama-layout decoders (also Cohere, Qwen2, Mistral).
_ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJ = ("gate_proj", "up_proj", "down_proj")


def circuit_info_from_node_scores(
    node_scores: Dict[str, float],
    *,
    top_fraction: float = 0.2,
    layers_path: str = "model.layers",
    attn_module: str = "self_attn",
    mlp_module: str = "mlp",
    attn_proj: tuple = _ATTN_PROJ,
    mlp_proj: tuple = _MLP_PROJ,
    raw_output_path: Optional[str] = None,
) -> CircuitInfo:
    """CircuitInfo from CircuitKIT ``node_scores``: the top ``top_fraction`` of nodes
    by score become ``safety_units`` (``activation_correlation`` = score / top score)
    and ``layer_suggestions`` (their layers, and the projections of the node kinds
    selected). Mirrors CircuitKIT 0.2's ``interop_fields``. Raises ``ValueError``
    when no top node has a recognisable name."""
    ranked = sorted(node_scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[: max(1, round(len(ranked) * top_fraction))] if ranked else []
    peak = max((v for _, v in top), default=0.0) or 1.0
    units, layers, kinds = [], set(), {}
    for name, score in top:
        m = _HEAD.match(name) or _MLP.match(name)
        if not m:
            continue
        kind = "attn" if m.re is _HEAD else "mlp"
        layer = int(m.group(1))
        module = attn_module if kind == "attn" else mlp_module
        units.append((name, f"{layers_path}.{layer}.{module}", score / peak))
        layers.add(layer)
        kinds[kind] = max(kinds.get(kind, 0.0), score / peak)
    if not units:
        raise ValueError(
            f"CircuitKIT node_scores{f' in {raw_output_path}' if raw_output_path else ''}: "
            f"none of the top {len(top)} of {len(ranked)} nodes has a recognised name "
            f"(A<layer>.<head>, 'MLP <layer>' or L<layer>.<neuron>); got {[n for n, _ in top][:5]}")
    priority = {p: w for kind, w in kinds.items()
                for p in (attn_proj if kind == "attn" else mlp_proj)}
    meta = {"source": "circuitkit node_scores", "top_fraction": top_fraction}
    return CircuitInfo(
        safety_units=SafetyRelevantUnits(
            layer_indices=sorted(layers),
            module_names=sorted({u[1] for u in units}),
            unit_ids=[u[0] for u in units],
            activation_correlation={u[0]: u[2] for u in units},
            metadata=dict(meta)),
        layer_suggestions=LayerModuleSuggestions(
            target_modules=list(priority), layer_subset=sorted(layers),
            priority=priority, metadata=dict(meta)),
        raw_output_path=raw_output_path,
    )


_SU_KEYS = {"layer_indices", "module_names", "unit_ids", "activation_correlation", "metadata"}
_LS_KEYS = {"target_modules", "layer_subset", "priority", "metadata"}


def _check_section(sec: Dict[str, Any], allowed: set, name: str, where: str,
                   int_lists: tuple, str_lists: tuple, num_maps: tuple) -> None:
    unknown = sorted(set(sec) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown} in {name!r} (allowed: {sorted(allowed)}).")
    for k in int_lists:
        v = sec.get(k)
        if v is None:
            continue
        if not isinstance(v, list) or not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in v):
            raise ValueError(f"{where}: {name}.{k} must be a list of non-negative integers, got {v!r}.")
    for k in str_lists:
        v = sec.get(k)
        if v is not None and (not isinstance(v, list) or not all(isinstance(i, str) for i in v)):
            raise ValueError(f"{where}: {name}.{k} must be a list of strings, got {v!r}.")
    for k in num_maps:
        v = sec.get(k)
        if v is not None and (not isinstance(v, dict) or not all(
                isinstance(a, str) and isinstance(b, (int, float)) and not isinstance(b, bool) for a, b in v.items())):
            raise ValueError(f"{where}: {name}.{k} must map strings to numbers, got {v!r}.")
    if "metadata" in sec and sec["metadata"] is not None and not isinstance(sec["metadata"], dict):
        raise ValueError(f"{where}: {name}.metadata must be a mapping.")


def _dict_to_circuit_info(data: Dict[str, Any], raw_output_path: Optional[str] = None,
                          top_fraction: float = 0.2) -> CircuitInfo:
    """Convert dict (e.g. from JSON/YAML) to CircuitInfo."""
    where = f"Circuit data {raw_output_path}" if raw_output_path else "Circuit data"
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be a mapping, got {type(data).__name__}.")
    if not data:
        return CircuitInfo(raw_output_path=raw_output_path)      # {} is an empty CircuitInfo
    for name in ("safety_units", "safety_relevant_units"):
        if isinstance(data.get(name), dict):
            _check_section(data[name], _SU_KEYS, name, where, ("layer_indices",),
                           ("module_names", "unit_ids"), ("activation_correlation",))
    for name in ("layer_suggestions", "lora_suggestions"):
        if isinstance(data.get(name), dict):
            _check_section(data[name], _LS_KEYS, name, where, ("layer_subset",), ("target_modules",), ("priority",))
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

    if safety_units is None and layer_suggestions is None:
        if data.get("node_scores"):
            info = circuit_info_from_node_scores(
                data["node_scores"], top_fraction=top_fraction, raw_output_path=raw_output_path)
            info.metadata = data.get("metadata", {})
            return info
        raise ValueError(
            f"No circuit information in {raw_output_path or 'the circuit data'}: expected "
            "'safety_units', 'layer_suggestions' or CircuitKIT 'node_scores'; "
            f"found keys {sorted(data) if isinstance(data, dict) else type(data).__name__}")

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
    Get circuit info from a file path.

    Only ``source`` and the ``top_fraction`` keyword are used. ``model_name`` and every other
    keyword are accepted and IGNORED (there is no API lookup); this function is not a
    :class:`CircuitInfoProvider` implementation.

    Args:
        source: Path to a CircuitKIT-format file, or None to return None.
        model_name: ignored.
        **kwargs: only ``top_fraction`` (for ``node_scores`` files) is read.

    Returns:
        CircuitInfo; ``None`` for no source or a missing file. An unreadable or invalid file
        raises ``ValueError`` (see :func:`load_circuit_info_from_file`).
    """
    if not source:
        return None
    # Warns and returns None when the file is missing; raises when it is unreadable or invalid.
    return load_circuit_info_from_file(source, top_fraction=kwargs.get("top_fraction", 0.2))


# ── Writer ──────────────────────────────────────────────────────────────────
# Inverse of the reader above. ``circuit_info_to_dict`` produces exactly the
# dict shape that ``_dict_to_circuit_info`` consumes, so
#   load_circuit_info_from_file -> save_circuit_info_to_file
# round-trips to an identical file. ``raw_output_path`` is intentionally NOT
# serialised: the reader derives it from the file path it was given, so writing
# it would break the round-trip. Nested ``metadata`` dicts are emitted only
# when non-empty, matching the reader's ``.get(..., {})`` defaults.


def _plain(obj: Any, where: str) -> Any:
    """Plain-Python copy of ``obj``: numpy / torch scalars and arrays become Python numbers and lists,
    tuples become lists, dict keys become strings; a non-finite float raises ``ValueError`` (JSON has no
    NaN, and a NaN score would read back as a real one)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v, f"{where}.{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v, f"{where}[{i}]") for i, v in enumerate(obj)]
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):      # numpy scalar / array, torch tensor
        return _plain(obj.tolist(), where)
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        raise ValueError(f"CircuitInfo contains a non-finite value ({obj}) at {where.lstrip('.')}; it cannot be saved.")
    return obj


def circuit_info_to_dict(info: CircuitInfo) -> Dict[str, Any]:
    """
    Serialise a :class:`CircuitInfo` to a plain dict.

    The result is the canonical (preferred-spelling) form accepted by
    :func:`_dict_to_circuit_info` / :func:`load_circuit_info_from_file`. numpy / torch scalars
    are cast to Python numbers, tuples become lists, keys become strings, and a non-finite value
    raises ``ValueError``.
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

    return _plain(data, "")


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
