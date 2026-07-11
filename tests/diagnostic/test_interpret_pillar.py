"""
Diagnostic suite for the Interpret pillar.

Two coverage goals:

1. ``identify_safety_neurons(method="weight")`` returns per-layer rankings
   whose top neurons have higher absolute cosine with the refusal direction
   than the bottom neurons.
2. The returned :class:`SafetyNeuronReport` converts cleanly to a
   :class:`CircuitInfo`, with unit ids in the expected format.

The synthetic model exposes ``mlp.down_proj`` and ``self_attn.o_proj`` in
the HF Llama convention so the ``target_module`` dotted-path resolution
exercises both.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class _LlamaBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.self_attn.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(hidden * 2, hidden, bias=False)

    def forward(self, x):
        return x


class _Wrap(nn.Module):
    def __init__(self, hidden: int = 32, n_layers: int = 4) -> None:
        super().__init__()
        self.model = nn.ModuleDict(
            {
                "embed_tokens": nn.Embedding(100, hidden),
                "layers": nn.ModuleList([_LlamaBlock(hidden) for _ in range(n_layers)]),
                "norm": nn.LayerNorm(hidden),
            }
        )
        self.lm_head = nn.Linear(hidden, 100, bias=False)

    def forward(self, x):
        return x


@pytest.fixture
def model_and_directions():
    """Construct a model whose layer-0 down_proj column 0 is aligned with the
    refusal direction (output-space axis). The last column is set to zero so
    it ranks last."""
    torch.manual_seed(0)
    hidden = 8
    model = _Wrap(hidden=hidden, n_layers=2)
    direction = torch.zeros(hidden)
    direction[3] = 1.0  # canonical axis in output space
    with torch.no_grad():
        # Column 0 of down_proj.weight is W[:, 0]; shape (hidden,) = (8,).
        # Set it to a vector parallel to the refusal direction.
        model.model["layers"][0].mlp.down_proj.weight[:, 0] = direction * 10.0
        # Last column zeroed -> excluded by score_floor=0.0.
        model.model["layers"][0].mlp.down_proj.weight[:, -1] = torch.zeros(hidden)
    return model, {0: direction, 1: direction}


def test_safety_neurons_weight_mode_ranks_aligned_first(model_and_directions):
    from safetune.core.interpret import SafetyNeuronConfig, identify_safety_neurons

    model, directions = model_and_directions
    cfg = SafetyNeuronConfig(method="weight", top_k_per_layer=4, target_module="mlp.down_proj")
    report = identify_safety_neurons(model, directions, cfg)
    assert 0 in report.per_layer
    top = report.per_layer[0]
    assert top, "report empty for layer 0"
    # The hand-aligned row index 0 should rank first.
    assert top[0][0] == 0, f"expected neuron 0 to rank first; got {top}"
    # The zero-column (last input neuron) must not be in the top-k.
    last_idx = model.model["layers"][0].mlp.down_proj.weight.shape[1] - 1
    assert all(i != last_idx for (i, _) in top), "zero-norm column should not be in top-k"


def test_safety_neurons_report_to_circuit_info(model_and_directions):
    from safetune.core.interpret import SafetyNeuronConfig, identify_safety_neurons

    model, directions = model_and_directions
    cfg = SafetyNeuronConfig(method="weight", top_k_per_layer=2)
    report = identify_safety_neurons(model, directions, cfg)
    info = report.as_circuit_info()
    assert info.safety_units is not None
    assert info.layer_suggestions is not None
    # Unit-id format: L<layer>.<module>.<idx>
    for uid in info.safety_units.unit_ids:
        assert uid.count(".") >= 2
        head, *_ = uid.split(".")
        assert head.startswith("L")
    # Layer suggestions inherits the same layer subset.
    assert set(info.layer_suggestions.layer_subset or []) == set(info.safety_units.layer_indices)


def test_safety_neurons_unknown_method_errors(model_and_directions):
    from safetune.core.interpret import SafetyNeuronConfig, identify_safety_neurons

    model, directions = model_and_directions
    with pytest.raises(ValueError):
        identify_safety_neurons(model, directions, SafetyNeuronConfig(method="bogus"))


def test_safety_neurons_target_layers_filter(model_and_directions):
    from safetune.core.interpret import SafetyNeuronConfig, identify_safety_neurons

    model, directions = model_and_directions
    cfg = SafetyNeuronConfig(top_k_per_layer=2, target_layers=[1])
    report = identify_safety_neurons(model, directions, cfg)
    assert set(report.per_layer.keys()) == {1}
