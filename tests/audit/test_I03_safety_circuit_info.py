"""Audit I03: safety_circuit_info -> CircuitInfo -> its readers (C-ΔΘ apply_ctheta reads layer_suggestions.layer_subset / target_modules).
Random tiny Llama; no network."""
import copy

import pytest
import torch

import safetune.recover as R
from safetune.core.circuit_kit.interface import CircuitInfo, LayerModuleSuggestions, SafetyRelevantUnits
from safetune.instrumentation.interpret import safety_circuit_info

from _interpret_common import HARMFUL, HARMLESS, build


def _pair(m):
    pos, neg = copy.deepcopy(m), copy.deepcopy(m)
    for p in pos.parameters():
        p.data += 0.01
    return pos, neg


def _changed(m, ci):
    target = copy.deepcopy(m)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    pos, neg = _pair(m)
    R.apply_ctheta(target, pos, neg, ci, strength=1.0)
    return {k for k, v in target.state_dict().items() if not torch.equal(v, before[k])}


def test_weight_mode_circuit_info_drives_ctheta_on_exactly_the_selected_layers_and_module():
    m, tok = build()
    ci = safety_circuit_info(m, tok, HARMFUL, HARMLESS, top_k_per_layer=3, target_layers=[1, 3], method="weight")
    assert ci.layer_suggestions.layer_subset == [1, 3] and ci.layer_suggestions.target_modules == ["mlp.down_proj"]
    assert _changed(m, ci) == {f"model.layers.{l}.mlp.down_proj.weight" for l in (1, 3)}      # 0-based, whole weight (unit ids are not read)
    assert ci.safety_units.metadata["direction_layer"] is not None


def test_activation_mode_circuit_info_can_be_consumed_by_ctheta():
    m, tok = build()
    ci = safety_circuit_info(m, tok, HARMFUL, HARMLESS, top_k_per_layer=3, target_layers=[0, 2], method="activation")
    assert ci.layer_suggestions.target_modules == ["mlp.down_proj"] and ci.layer_suggestions.layer_subset == [0, 2]
    assert _changed(m, ci) == {f"model.layers.{l}.mlp.down_proj.weight" for l in (0, 2)}


def test_activation_mode_does_not_extract_a_refusal_direction(monkeypatch):
    import safetune.interventions.steer.refusal_direction as RD

    def boom(*a, **k):
        raise AssertionError("extract_refusal_direction was called in activation mode")
    monkeypatch.setattr(RD, "extract_refusal_direction", boom)
    m, tok = build()
    ci = safety_circuit_info(m, tok, HARMFUL, HARMLESS, top_k_per_layer=2, method="activation")
    assert ci.safety_units.metadata["direction_layer"] is None
    with pytest.raises(AssertionError, match="extract_refusal_direction"):
        safety_circuit_info(m, tok, HARMFUL, HARMLESS, top_k_per_layer=2, method="weight")


def test_an_empty_circuit_info_is_refused_by_ctheta():
    m, _ = build()
    pos, neg = _pair(m)
    empty = CircuitInfo(SafetyRelevantUnits(), LayerModuleSuggestions(target_modules=["mlp.down_proj"], layer_subset=[]))
    with pytest.raises(ValueError, match="no parameters matched"):
        R.apply_ctheta(copy.deepcopy(m), pos, neg, empty)
