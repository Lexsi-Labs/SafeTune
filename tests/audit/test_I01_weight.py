"""Audit I01 (weight mode). Random tiny Llama; placeholder data. The cosine ranking is SafeTune's own heuristic (no paper defines it)."""
import copy

import pytest
import torch

from safetune.instrumentation.interpret import SafetyNeuronConfig, identify_safety_neurons

from _interpret_common import build


def _dirs(m, seed=1):
    torch.manual_seed(seed)
    d = torch.randn(m.config.hidden_size)
    return d, {l: d for l in range(m.config.num_hidden_layers)}


def test_weight_mode_ranks_columns_by_absolute_cosine_bit_identically():
    m, _ = build()
    d, dirs = _dirs(m)
    rep = identify_safety_neurons(m, dirs, SafetyNeuronConfig(method="weight", top_k_per_layer=3))
    for layer in range(4):
        W = m.model.layers[layer].mlp.down_proj.weight
        cos = (d / d.norm()) @ W / W.norm(dim=0)
        assert [i for i, _ in rep.per_layer[layer]] == cos.abs().topk(3).indices.tolist()
        assert all(abs(s - abs(float(cos[i]))) < 1e-5 for i, s in rep.per_layer[layer])


def test_target_layers_filter_and_other_module_still_work():
    m, _ = build()
    _, dirs = _dirs(m)
    rep = identify_safety_neurons(m, dirs, SafetyNeuronConfig(top_k_per_layer=2, target_layers=[1, 3]))
    assert sorted(rep.per_layer) == [1, 3]
    rep = identify_safety_neurons(m, dirs, SafetyNeuronConfig(top_k_per_layer=2, target_module="self_attn.o_proj"))
    assert sorted(rep.per_layer) == [0, 1, 2, 3] and rep.target_module == "self_attn.o_proj"


@pytest.mark.parametrize("case,pattern", [("wrong_dim", "width 7"), ("layer_out_of_range", "out of range"),
                                          ("lora_wrapped", "nn.Linear"), ("score_floor_too_high", "scored no unit")])
def test_a_selection_that_scores_nothing_raises(case, pattern):
    m, _ = build()
    d, dirs = _dirs(m)
    cfg = SafetyNeuronConfig(method="weight", top_k_per_layer=2)
    if case == "wrong_dim":
        dirs = {l: torch.randn(7) for l in dirs}
    elif case == "layer_out_of_range":
        cfg.target_layers = [99]
    elif case == "score_floor_too_high":
        cfg.score_floor = 5.0
    else:
        peft = pytest.importorskip("peft")
        m = peft.get_peft_model(copy.deepcopy(m), peft.LoraConfig(r=4, target_modules=["down_proj"]))
    with pytest.raises(ValueError, match=pattern):
        identify_safety_neurons(m, dirs, cfg)


def test_activation_mode_with_target_layers_out_of_range_raises_too():
    from _interpret_common import HARMFUL, HARMLESS
    m, tok = build()
    cfg = SafetyNeuronConfig(method="activation", target_layers=[99])
    with pytest.raises(ValueError, match="out of range"):
        identify_safety_neurons(m, {}, cfg, tokenizer=tok, harmful_prompts=HARMFUL, harmless_prompts=HARMLESS)


def test_a_layer_without_a_direction_raises_instead_of_borrowing_another_layers():
    m, _ = build()
    d, _ = _dirs(m)
    with pytest.raises(ValueError, match="no refusal direction for layer 1"):
        identify_safety_neurons(m, {0: d}, SafetyNeuronConfig(method="weight", top_k_per_layer=2))
    rep = identify_safety_neurons(m, {0: d}, SafetyNeuronConfig(method="weight", top_k_per_layer=2, target_layers=[0]))
    assert sorted(rep.per_layer) == [0]                                   # restricting to the layers you have is fine


def test_a_bare_tensor_is_not_a_dict_of_directions():
    m, _ = build()
    d, _ = _dirs(m)
    with pytest.raises((ValueError, AttributeError, TypeError)):
        identify_safety_neurons(m, d, SafetyNeuronConfig(method="weight"))


@pytest.mark.parametrize("k", [0, -1, -5])
def test_top_k_per_layer_must_be_positive(k):
    m, _ = build()
    _, dirs = _dirs(m)
    with pytest.raises(ValueError, match="top_k_per_layer"):
        identify_safety_neurons(m, dirs, SafetyNeuronConfig(top_k_per_layer=k, target_layers=[0]))


def test_top_k_larger_than_the_number_of_neurons_keeps_them_all():
    m, _ = build()
    _, dirs = _dirs(m)
    rep = identify_safety_neurons(m, dirs, SafetyNeuronConfig(top_k_per_layer=10_000, target_layers=[0]))
    assert len(rep.per_layer[0]) == m.config.intermediate_size
