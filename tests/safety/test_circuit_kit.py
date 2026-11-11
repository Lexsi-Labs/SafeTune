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


# 36 layers of 16 heads and an MLP (612 nodes, so the top 20% is 122). Seven layers hold
# nearly all the score; 23 others hold one node just inside the top 20%; 6 hold none.
STRONG_LAYERS = [3, 8, 13, 18, 23, 28, 33]
QUIET_LAYERS = [0, 1, 2, 32, 34, 35]


def _spread_scores():
    scores = {}
    for layer in range(36):
        names = [f"A{layer}.{head}" for head in range(16)] + [f"MLP {layer}"]
        strong = layer in STRONG_LAYERS
        hot = int(layer not in QUIET_LAYERS)
        if strong:
            hot = 15 if layer == STRONG_LAYERS[0] else 14
        for rank, name in enumerate(names):
            scores[name] = 0.01 if rank >= hot else 1.0 if strong else 0.9
    return scores


def test_default_keeps_every_layer_holding_a_top_node():
    from safetune.core.circuit_kit import circuit_info_from_node_scores

    info = circuit_info_from_node_scores(_spread_scores())
    assert len(info.safety_units.unit_ids) == 122
    assert info.layer_suggestions.layer_subset == [k for k in range(36) if k not in QUIET_LAYERS]
    assert "layer_fraction" not in info.layer_suggestions.metadata


def test_layer_fraction_ranks_layers_by_total_score():
    from safetune.core.circuit_kit import circuit_info_from_node_scores

    info = circuit_info_from_node_scores(_spread_scores(), layer_fraction=0.2)
    assert info.layer_suggestions.layer_subset == STRONG_LAYERS
    assert info.safety_units.layer_indices == STRONG_LAYERS
    assert {int(m.split(".")[2]) for m in info.safety_units.module_names} <= set(STRONG_LAYERS)
    assert info.layer_suggestions.metadata["layer_fraction"] == 0.2


def test_a_top_node_outside_the_ranked_layers_leaves_units_to_pick_from():
    from safetune.core.circuit_kit import circuit_info_from_node_scores

    scores = {"A0.0": 10.0, "A1.0": 6.0, "A1.1": 6.0}  # layer 1 outweighs the top node
    info = circuit_info_from_node_scores(scores, top_fraction=0.34, layer_fraction=0.5)
    assert info.layer_suggestions.layer_subset == [1]
    assert info.safety_units.unit_ids == ["A1.0"]


def test_layer_mass_adds_absolute_scores_and_skips_nodes_without_a_layer():
    from safetune.core.circuit_kit import layer_mass

    scores = {"A0.0": -2.0, "A0.1": 1.0, "MLP 1": 1.5, "L1.7": 0.5, "input": 9.0, "logits": 9.0}
    assert layer_mass(scores) == {0: 3.0, 1: 2.0}


def test_top_layers_keeps_at_least_one_and_breaks_ties_low():
    from safetune.core.circuit_kit import top_layers

    assert top_layers({"A0.0": 1.0, "A1.0": 1.0}, 0.01) == [0]
    assert top_layers({"A0.0": 1.0, "A1.0": 2.0, "A2.0": 3.0}, 1.0) == [0, 1, 2]


def test_top_layers_never_selects_a_layer_with_no_score():
    from safetune.core.circuit_kit import top_layers

    assert top_layers({"A0.0": 1.0, "A1.0": 0.0}, 1.0) == [0]


@pytest.mark.parametrize("fraction", [0, -0.1, 1.5, float("nan")])
def test_top_layers_rejects_a_fraction_outside_the_range(fraction):
    from safetune.core.circuit_kit import top_layers

    with pytest.raises(ValueError, match="layer_fraction"):
        top_layers({"A0.0": 1.0}, fraction)


@pytest.mark.parametrize(
    "scores,match",
    [
        ({"A0.0": 0.0}, "every score is zero"),
        ({"A0.0": float("nan")}, "has the score"),
        ({"blocks.0.hook_z": 1.0}, "recognised name"),
        ({}, "recognised name"),
    ],
)
def test_top_layers_refuses_scores_it_cannot_rank(scores, match):
    from safetune.core.circuit_kit import top_layers

    with pytest.raises(ValueError, match=match):
        top_layers(scores, 0.5)


def _write(tmp_path, data):
    path = tmp_path / "x_scores.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_layer_fraction_replaces_the_keys_circuitkit_02_writes(tmp_path):
    """CircuitKIT 0.2 stores the unranked selection, so a real file must not win here."""
    from safetune.core.circuit_kit import get_circuit_info, load_circuit_info_from_file

    path = _write(
        tmp_path,
        {
            "node_scores": _spread_scores(),
            "layer_suggestions": {"target_modules": ["q_proj"], "layer_subset": list(range(30))},
        },
    )
    assert load_circuit_info_from_file(path).layer_suggestions.layer_subset == list(range(30))
    assert (
        load_circuit_info_from_file(path, layer_fraction=0.2).layer_suggestions.layer_subset
        == STRONG_LAYERS
    )
    ranked = get_circuit_info(path, layer_fraction=0.2)
    assert ranked.layer_suggestions.layer_subset == STRONG_LAYERS


def test_layer_fraction_without_node_scores_warns_and_keeps_the_file(tmp_path, caplog):
    from safetune.core.circuit_kit import load_circuit_info_from_file

    path = _write(
        tmp_path, {"layer_suggestions": {"target_modules": ["q_proj"], "layer_subset": [3]}}
    )
    with caplog.at_level("WARNING", logger="safetune.core.circuit_kit.adapter"):
        info = load_circuit_info_from_file(path, layer_fraction=0.2)
    assert info.layer_suggestions.layer_subset == [3]
    assert "layer_fraction ignored" in caplog.text


def test_ranked_selection_survives_a_save_and_load(tmp_path):
    from safetune.core.circuit_kit import (
        circuit_info_from_node_scores,
        load_circuit_info_from_file,
        save_circuit_info_to_file,
    )

    path = tmp_path / "circuit.json"
    save_circuit_info_to_file(
        circuit_info_from_node_scores(_spread_scores(), layer_fraction=0.2), str(path)
    )
    assert load_circuit_info_from_file(str(path)).layer_suggestions.layer_subset == STRONG_LAYERS
