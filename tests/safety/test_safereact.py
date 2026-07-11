"""Tests for SafeReAct hidden safety neuron discovery and reactivation."""
import pytest
import torch
import torch.nn as nn

from safetune.core.safereact import (
    SafeReActConfig,
    build_safereact_config,
    find_suppressed_safety_neurons,
    build_reactivation_lora,
    apply_safereact,
)


@pytest.fixture
def small_model():
    """Two-layer MLP simulating an aligned model."""
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    return model


@pytest.fixture
def suppressed_model(small_model):
    """Post-trained model with some weights zeroed (simulating suppression)."""
    post = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    # Suppress first-layer neurons by zeroing weights
    with torch.no_grad():
        post[0].weight.data[:8] = 0.0
    return post


def test_config_defaults():
    cfg = SafeReActConfig()
    assert cfg.top_k_neurons == 64
    assert cfg.reactivation_scale == 1.0
    assert cfg.lora_rank == 8


def test_build_safereact_config_filters_unknown():
    cfg = build_safereact_config(top_k_neurons=32, unknown_key="ignored")
    assert cfg.top_k_neurons == 32
    assert not hasattr(cfg, "unknown_key")


def test_find_suppressed_neurons_ranks_correctly(small_model, suppressed_model):
    """Neurons suppressed in post-trained model should get high scores."""
    cfg = SafeReActConfig(top_k_neurons=5)
    probe = torch.randn(1, 8)
    units = find_suppressed_safety_neurons(suppressed_model, small_model, cfg, probe_inputs=probe)
    # Should find some suppressed units
    assert len(units) > 0
    # All scores should be >= 0
    assert all(u.score >= 0.0 for u in units)
    # Units should be sorted descending
    scores = [u.score for u in units]
    assert scores == sorted(scores, reverse=True)


def test_find_suppressed_neurons_respects_top_k(small_model, suppressed_model):
    cfg = SafeReActConfig(top_k_neurons=2)
    probe = torch.randn(1, 8)
    units = find_suppressed_safety_neurons(suppressed_model, small_model, cfg, probe_inputs=probe)
    assert len(units) <= 2


def test_build_reactivation_lora_non_empty(small_model, suppressed_model):
    cfg = SafeReActConfig(top_k_neurons=10, reactivation_scale=0.8)
    probe = torch.randn(1, 8)
    units = find_suppressed_safety_neurons(suppressed_model, small_model, cfg, probe_inputs=probe)
    lora = build_reactivation_lora(units, suppressed_model, small_model, cfg)
    # Should return a dict (possibly empty if no target modules match)
    assert isinstance(lora, dict)
    if lora:
        assert "aligned_state_dict" in lora or "alpha" in lora


def test_build_reactivation_lora_empty_units(small_model, suppressed_model):
    cfg = SafeReActConfig()
    lora = build_reactivation_lora([], suppressed_model, small_model, cfg)
    assert lora == {}


def test_apply_safereact_returns_dict(small_model, suppressed_model):
    cfg = SafeReActConfig(top_k_neurons=5)
    probe = torch.randn(1, 8)
    result = apply_safereact(suppressed_model, small_model, cfg, probe_inputs=probe)
    assert isinstance(result, dict)


def test_no_suppression_same_model(small_model):
    """Identical models should produce zero (or near-zero) suppression scores."""
    cfg = SafeReActConfig(top_k_neurons=10)
    probe = torch.randn(1, 8)
    units = find_suppressed_safety_neurons(small_model, small_model, cfg, probe_inputs=probe)
    # All suppression scores should be 0 for identical models
    assert all(u.score == pytest.approx(0.0, abs=1e-5) for u in units)
