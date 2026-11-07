"""Audit I05: save_circuit_info_to_file / CircuitInfo.to_dict and the provider. Pure Python (torch only for the scalar case)."""
import numpy as np
import pytest
import torch

from safetune.core.circuit_kit import (CircuitInfo, CircuitInfoProvider, circuit_info_to_dict, get_circuit_info,
                                       load_circuit_info_from_file, save_circuit_info_to_file)
from safetune.core.circuit_kit.interface import LayerModuleSuggestions, SafetyRelevantUnits


@pytest.mark.parametrize("ext", ["json", "yaml"])
def test_numpy_and_torch_scalars_are_saved_as_plain_numbers(tmp_path, ext):
    info = CircuitInfo(SafetyRelevantUnits([np.int64(1), torch.tensor(3)], ["m"], ["u", "v"],
                                           {"u": np.float32(0.5), "v": torch.tensor(0.25)}, {"k": (np.int32(2), 3)}),
                       LayerModuleSuggestions(["m"], np.array([1, 3]), {"m": np.float64(0.75)}))
    p = tmp_path / f"np.{ext}"
    save_circuit_info_to_file(info, str(p))
    back = load_circuit_info_from_file(str(p))
    assert back.safety_units.layer_indices == [1, 3] and back.safety_units.activation_correlation == {"u": 0.5, "v": 0.25}
    assert back.safety_units.metadata == {"k": [2, 3]} and back.layer_suggestions.layer_subset == [1, 3]
    assert back.layer_suggestions.priority == {"m": 0.75}
    d = circuit_info_to_dict(info)
    assert type(d["safety_units"]["layer_indices"][0]) is int and type(d["safety_units"]["activation_correlation"]["u"]) is float


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), np.float32("nan")])
def test_a_non_finite_value_is_refused_and_nothing_is_written(tmp_path, bad):
    p = tmp_path / "n.json"
    info = CircuitInfo(SafetyRelevantUnits([1], ["m"], ["u"], {"u": bad}, {}), None)
    with pytest.raises(ValueError, match="non-finite.*activation_correlation.u"):
        save_circuit_info_to_file(info, str(p))
    assert not p.exists()
    with pytest.raises(ValueError, match="non-finite"):
        info.to_dict()


def test_finite_python_values_are_written_exactly_as_before(tmp_path):
    info = CircuitInfo(SafetyRelevantUnits([0, 2], ["mlp.down_proj"], ["a", "b"], {"a": 0.25, "b": 1e-05, "c": -3.0}, {"method": "x"}),
                       LayerModuleSuggestions(["mlp.down_proj"], [0, 2], {"mlp.down_proj": 0.5}), None, {"a": 1})
    p = tmp_path / "c.json"
    save_circuit_info_to_file(info, str(p))
    text = p.read_text()
    assert '"b": 1e-05' in text and text.endswith("}\n") and list(circuit_info_to_dict(info)) == ["safety_units", "layer_suggestions", "metadata"]
    back = load_circuit_info_from_file(str(p))
    assert circuit_info_to_dict(back) == circuit_info_to_dict(info)


def test_the_provider_is_an_unimplemented_protocol_and_get_circuit_info_ignores_model_name(tmp_path):
    assert "Protocol" in " ".join(CircuitInfoProvider.__doc__.split()) and "nothing in SafeTune implements it" in " ".join(CircuitInfoProvider.__doc__.split())
    doc = " ".join(get_circuit_info.__doc__.split())
    assert "IGNORED" in doc and "not a :class:`CircuitInfoProvider` implementation" in doc
    p = tmp_path / "c.json"
    save_circuit_info_to_file(CircuitInfo(SafetyRelevantUnits([1], ["m"], ["u"], {}, {}), None), str(p))
    a = get_circuit_info(str(p))
    b = get_circuit_info(str(p), model_name="anything", whatever=1)
    assert circuit_info_to_dict(a) == circuit_info_to_dict(b)
