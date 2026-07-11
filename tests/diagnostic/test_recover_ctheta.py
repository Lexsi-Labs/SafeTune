"""Behavioral tests for C-ΔΘ circuit-guided weight steering.

Three guarantees the implementation must give:
  1. Only parameters that pass the (layer_subset, target_modules) mask
     are mutated; off-circuit parameters stay bit-identical.
  2. The mutation direction equals ``strength * (θ_positive - θ_negative)``.
  3. Empty mask raises clearly, rather than silently no-op'ing.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn


class _ToyBlock(nn.Module):
    """Mimics one transformer block with q/k/v/o projections."""
    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.self_attn_q_proj = nn.Linear(hidden, hidden, bias=False)
        self.self_attn_k_proj = nn.Linear(hidden, hidden, bias=False)
        self.self_attn_v_proj = nn.Linear(hidden, hidden, bias=False)
        self.self_attn_o_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_up_proj = nn.Linear(hidden, 2 * hidden, bias=False)
        self.mlp_down_proj = nn.Linear(2 * hidden, hidden, bias=False)


class _ToyModel(nn.Module):
    """4-layer toy transformer with the HF naming convention SafeTune expects."""
    def __init__(self, n_layers: int = 4, hidden: int = 8) -> None:
        super().__init__()
        self.model_layers = nn.ModuleList([_ToyBlock(hidden) for _ in range(n_layers)])

    def state_dict_hf_named(self) -> dict:
        """Re-key state_dict with HF-style keys (`model.layers.<i>.q_proj.weight`)."""
        sd = {}
        for i, block in enumerate(self.model_layers):
            for sub, mod in block.named_children():
                # sub is "self_attn_q_proj" -> "self_attn.q_proj"
                hf_sub = sub.replace("self_attn_", "self_attn.").replace("mlp_", "mlp.")
                sd[f"model.layers.{i}.{hf_sub}.weight"] = mod.weight.detach().clone()
        return sd


def _make_hf_state_dict(model: _ToyModel, *, randomize: bool, seed: int) -> dict:
    """Generate a fresh HF-shaped state_dict for the toy model with new weights."""
    torch.manual_seed(seed)
    fresh = _ToyModel(n_layers=len(model.model_layers), hidden=model.model_layers[0].self_attn_q_proj.in_features)
    if not randomize:
        for p in fresh.parameters():
            p.data.zero_()
    return fresh.state_dict_hf_named()


class _HFShapedModel(nn.Module):
    """Wrapper whose .state_dict() returns HF-keyed tensors so apply_ctheta works."""
    def __init__(self, hf_sd: dict) -> None:
        super().__init__()
        for k, v in hf_sd.items():
            self.register_buffer(k.replace(".", "__"), v.clone())
        self._key_map = {k.replace(".", "__"): k for k in hf_sd}

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        out = {}
        for safe, hf in self._key_map.items():
            out[hf] = getattr(self, safe).clone()
        return out

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        for hf, v in state_dict.items():
            safe = hf.replace(".", "__")
            if hasattr(self, safe):
                getattr(self, safe).data.copy_(v)


def _build_triple(hidden: int = 8, n_layers: int = 4):
    sd_target = _make_hf_state_dict(_ToyModel(n_layers, hidden), randomize=True, seed=1)
    sd_positive = _make_hf_state_dict(_ToyModel(n_layers, hidden), randomize=True, seed=2)
    sd_negative = _make_hf_state_dict(_ToyModel(n_layers, hidden), randomize=True, seed=3)
    return (_HFShapedModel(sd_target), _HFShapedModel(sd_positive), _HFShapedModel(sd_negative))


def _circuit_info(layer_subset=None, target_modules=None):
    from safetune.core.circuit_kit import CircuitInfo, LayerModuleSuggestions
    return CircuitInfo(
        layer_suggestions=LayerModuleSuggestions(
            target_modules=list(target_modules or []),
            layer_subset=list(layer_subset) if layer_subset is not None else None,
        ),
    )


def test_ctheta_only_mutates_masked_parameters():
    """Off-circuit keys must stay bit-identical."""
    from safetune.recover import apply_ctheta

    target, positive, negative = _build_triple()
    before = {k: v.clone() for k, v in target.state_dict().items()}

    info = _circuit_info(layer_subset=[1, 2], target_modules=["q_proj", "v_proj"])
    apply_ctheta(target, positive, negative, info, strength=1.0)
    after = target.state_dict()

    for k in before:
        layer_idx = int(k.split(".")[2])
        is_in_layers = layer_idx in {1, 2}
        is_in_modules = any(m in k for m in ("q_proj", "v_proj"))
        in_circuit = is_in_layers and is_in_modules
        if in_circuit:
            assert not torch.allclose(before[k], after[k]), \
                f"in-circuit key {k} was not mutated"
        else:
            assert torch.allclose(before[k], after[k]), \
                f"off-circuit key {k} was mutated"


def test_ctheta_direction_matches_delta():
    """θ_after − θ_before == strength * (θ_pos − θ_neg) on matched keys."""
    from safetune.recover import apply_ctheta

    target, positive, negative = _build_triple()
    before = {k: v.clone() for k, v in target.state_dict().items()}
    sd_p = positive.state_dict()
    sd_n = negative.state_dict()

    info = _circuit_info(layer_subset=[0], target_modules=["q_proj"])
    apply_ctheta(target, positive, negative, info, strength=0.5)
    after = target.state_dict()

    matched = [k for k in before if k.startswith("model.layers.0.") and "q_proj" in k]
    assert matched, "expected at least one matched parameter"
    for k in matched:
        expected = before[k] + 0.5 * (sd_p[k] - sd_n[k])
        assert torch.allclose(after[k], expected, atol=1e-6), \
            f"steering direction wrong for {k}"


def test_ctheta_empty_mask_raises():
    """A mask that selects nothing should raise, not silently no-op."""
    from safetune.recover import apply_ctheta

    target, positive, negative = _build_triple()
    info = _circuit_info(layer_subset=[99], target_modules=["q_proj"])
    with pytest.raises(ValueError):
        apply_ctheta(target, positive, negative, info, strength=1.0)


def test_ctheta_sweep_explores_strengths_and_restores_target():
    """sweep_ctheta_strength must:
       1. call eval_fn once per strength,
       2. mark the best strength,
       3. leave the target bit-identical at the end.
    """
    from safetune.recover import sweep_ctheta_strength

    target, positive, negative = _build_triple()
    snapshot = {k: v.clone() for k, v in target.state_dict().items()}

    calls = []
    def _fake_eval(model):
        sd = model.state_dict()
        # Score = mean abs delta on a known masked key (proxy for ASR).
        key = "model.layers.0.self_attn.q_proj.weight"
        calls.append(float(sd[key].abs().mean().item()))
        return calls[-1]

    info = _circuit_info(layer_subset=[0], target_modules=["q_proj"])
    rows = sweep_ctheta_strength(
        target, positive, negative, info,
        strengths=[0.5, 1.0, 2.0],
        eval_fn=_fake_eval,
        higher_is_better=False,
    )

    assert [r["strength"] for r in rows] == [0.5, 1.0, 2.0]
    assert len(calls) == 3
    assert sum(int(r.get("best", False)) for r in rows) == 1

    # Target must be restored to original weights after the sweep.
    final_sd = target.state_dict()
    for k, v in snapshot.items():
        assert torch.allclose(v, final_sd[k], atol=1e-5), \
            f"target not restored after sweep at {k}"


def test_ctheta_from_state_dicts_skips_reload():
    """In-memory variant: same direction guarantee without holding two extra models."""
    from safetune.recover import apply_ctheta_from_state_dicts

    target, positive, negative = _build_triple()
    sd_p = positive.state_dict()
    sd_n = negative.state_dict()
    before = {k: v.clone() for k, v in target.state_dict().items()}

    info = _circuit_info(layer_subset=[3], target_modules=["mlp.up_proj"])
    apply_ctheta_from_state_dicts(target, sd_p, sd_n, info, strength=2.0)
    after = target.state_dict()

    matched = [k for k in before if k.startswith("model.layers.3.") and "mlp.up_proj" in k]
    assert matched
    for k in matched:
        expected = before[k] + 2.0 * (sd_p[k] - sd_n[k])
        assert torch.allclose(after[k], expected, atol=1e-6)
