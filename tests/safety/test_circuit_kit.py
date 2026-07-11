"""Tests for CircuitKIT adapter."""
import json
import tempfile
from pathlib import Path
import pytest


def test_adapter_load_json():
    from safetune.core.circuit_kit import load_circuit_info_from_file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"layer_suggestions": {"target_modules": ["q_proj", "v_proj"]}}, f)
        path = f.name
    try:
        info = load_circuit_info_from_file(path)
        assert info is not None
        assert info.layer_suggestions.target_modules == ["q_proj", "v_proj"]
    finally:
        Path(path).unlink(missing_ok=True)


def test_get_circuit_info_none():
    from safetune.core.circuit_kit import get_circuit_info
    assert get_circuit_info(source=None) is None
