"""``lexsi_provenance.json``: the provenance record shared by the Lexsi libraries.

Schema ``lexsi.provenance/1``. SafeTune writes it into every checkpoint, results and
steering-vector directory it creates. When an input (a CuratorKIT export folder, a
model folder written by SafeTune / AlignTune / CircuitKIT) carries its own
``lexsi_provenance.json``, that record is nested under ``inputs[].provenance`` so
lineage chains across the stack. Readers tolerate a missing or unreadable file.
"""
from __future__ import annotations

import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

FILENAME = "lexsi_provenance.json"
SCHEMA = "lexsi.provenance/1"


def read_provenance(path: Any) -> Optional[dict]:
    """The record next to ``path`` (a folder, or a file's folder); ``None`` for Hub ids,
    a missing file or bad JSON."""
    try:
        p = Path(path)
        if not p.exists():  # a Hub id or table name, not ./lexsi_provenance.json
            return None
        return json.loads(((p if p.is_dir() else p.parent) / FILENAME).read_text())
    except (OSError, TypeError, ValueError):
        return None


def input_entry(kind: str, ref: Any, config: Optional[str] = None) -> dict:
    """One ``inputs[]`` entry, carrying the parent's record when it has one."""
    entry = {"kind": kind, "ref": None if ref is None else str(ref)}
    if kind == "dataset":
        entry["config"] = config
    entry["provenance"] = read_provenance(ref) if ref else None
    return entry


def make_provenance(method: Optional[str], *, inputs: Optional[list] = None,
                    base_model: Optional[str] = None, params: Optional[dict] = None) -> dict:
    """A ``lexsi.provenance/1`` record. ``base_model`` defaults to the lineage of the
    first model input (its own ``base_model``, else its ref)."""
    inputs = list(inputs or [])
    if base_model is None:
        model = next((i for i in inputs if i.get("kind") == "model"), None)
        if model:
            base_model = (model.get("provenance") or {}).get("base_model") or model.get("ref")
    try:
        version = importlib.metadata.version("safetune")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "schema": SCHEMA,
        "library": "safetune",
        "version": version,
        "git_sha": None,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_model": base_model,
        "method": method,
        "inputs": inputs,
        "params": dict(params or {}),
    }


def write_provenance(output_dir: Any, method: Optional[str], **kw: Any) -> dict:
    """Write ``make_provenance(method, **kw)`` to ``<output_dir>/lexsi_provenance.json``."""
    record = make_provenance(method, **kw)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / FILENAME).write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record
