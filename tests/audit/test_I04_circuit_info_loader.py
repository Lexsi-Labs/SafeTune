"""Audit I04: CircuitInfo JSON/YAML reader. Pure Python; no model."""
import pytest

from safetune.core.circuit_kit import (CircuitInfo, circuit_info_to_dict, get_circuit_info, load_circuit_info_from_file,
                                       save_circuit_info_to_file)
from safetune.core.circuit_kit.interface import LayerModuleSuggestions, SafetyRelevantUnits


def _info():
    return CircuitInfo(
        SafetyRelevantUnits([0, 2], ["mlp.down_proj"], ["L0.mlp.down_proj.5", "L2.mlp.down_proj.9"],
                            {"L0.mlp.down_proj.5": 0.25, "L2.mlp.down_proj.9": 1e-05}, {"method": "x", "k": [1, 2]}),
        LayerModuleSuggestions(["mlp.down_proj"], [0, 2], {"mlp.down_proj": 0.5}, {"s": "y"}), None, {"a": 1})


@pytest.mark.parametrize("ext", ["json", "yaml", "yml"])
def test_round_trip_is_exact_and_resave_is_byte_identical(tmp_path, ext):
    p = tmp_path / f"c.{ext}"
    save_circuit_info_to_file(_info(), str(p))
    back = load_circuit_info_from_file(str(p))
    assert circuit_info_to_dict(back) == circuit_info_to_dict(_info())
    text = p.read_text()
    save_circuit_info_to_file(back, str(p))
    assert p.read_text() == text and back.raw_output_path == str(p)


def test_a_missing_file_and_no_source_still_return_none(tmp_path):
    assert load_circuit_info_from_file(str(tmp_path / "nope.json")) is None and get_circuit_info(None) is None
    assert get_circuit_info(str(tmp_path / "nope.yaml")) is None


@pytest.mark.parametrize("name,text,why", [("bad.json", "{not json", "JSONDecodeError"), ("bad.yaml", "a: [1, 2", "YAMLError|ParserError"),
                                           ("empty.yaml", "", "empty file"), ("list.json", "[1, 2]", "list"), ("scalar.yaml", "5", "int")])
def test_an_unreadable_or_non_mapping_file_raises_valueerror_with_the_path_and_reason(tmp_path, name, text, why):
    p = tmp_path / name
    p.write_text(text)
    with pytest.raises(ValueError, match=name) as e:
        load_circuit_info_from_file(str(p))
    import re
    assert re.search(why, str(e.value))
    with pytest.raises(ValueError, match=name):
        get_circuit_info(str(p))


def test_a_directory_is_not_a_circuit_file(tmp_path):
    with pytest.raises(ValueError, match="Cannot read circuit file"):
        load_circuit_info_from_file(str(tmp_path))


@pytest.mark.parametrize("body,needle", [
    ('{"safety_units": {"layer_indicies": [1, 2], "module_names": ["x"]}}', "layer_indicies"),
    ('{"layer_suggestions": {"target_module": ["q_proj"]}}', "target_module"),
    ('{"safety_units": {"layer_indices": ["a", -3, 1.5]}}', "layer_indices"),
    ('{"safety_units": {"layer_indices": [-1]}}', "layer_indices"),
    ('{"safety_units": {"layer_indices": [true]}}', "layer_indices"),
    ('{"layer_suggestions": {"layer_subset": [1.5]}}', "layer_subset"),
    ('{"safety_units": {"unit_ids": [1]}}', "unit_ids"),
    ('{"safety_units": {"activation_correlation": {"u": "high"}}}', "activation_correlation"),
])
def test_unknown_nested_keys_and_bad_layer_indices_raise(tmp_path, body, needle):
    p = tmp_path / "c.json"
    p.write_text(body)
    with pytest.raises(ValueError, match=needle):
        load_circuit_info_from_file(str(p))


def test_an_empty_mapping_is_an_empty_circuit_info_and_unknown_top_level_files_still_raise(tmp_path):
    p = tmp_path / "e.json"
    p.write_text("{}")
    ci = load_circuit_info_from_file(str(p))
    assert ci.safety_units is None and ci.layer_suggestions is None and ci.metadata == {}
    p.write_text('{"foo": 1}')
    with pytest.raises(ValueError, match="No circuit information"):
        load_circuit_info_from_file(str(p))


def test_circuitkit_node_scores_and_both_spellings_still_load(tmp_path):
    p = tmp_path / "n.json"
    p.write_text('{"node_scores": {"A3.5": 2.0, "MLP 3": 1.0, "A1.0": 0.1}}')
    ci = load_circuit_info_from_file(str(p), top_fraction=0.7)
    assert ci.layer_suggestions.layer_subset == [3]
    p.write_text('{"safety_relevant_units": {"layer_indices": [4]}, "lora_suggestions": {"target_modules": ["q_proj"], "layer_subset": [4]}}')
    ci = load_circuit_info_from_file(str(p))
    assert ci.safety_units.layer_indices == [4] and ci.layer_suggestions.layer_subset == [4]
