"""Tests for AAQ Alignment-Aware Quantization patch."""
import pytest
import torch
import torch.nn as nn

from safetune.core.patches.aaq_patch import AAQPatch, _apc_loss, AAQConfig


@pytest.fixture
def tiny_model():
    return nn.Linear(8, 8)


def test_aaq_config_defaults():
    cfg = AAQConfig()
    assert cfg.quantization_bits == 4
    assert cfg.apc_weight == 0.1
    assert cfg.calibration_steps == 20


def test_apc_loss_aligned_closer():
    """APC loss is lower when quantized is closer to aligned than base."""
    torch.manual_seed(42)
    base   = torch.zeros(1, 32)
    aligned = torch.ones(1, 32)
    quant_close   = torch.ones(1, 32) * 0.9   # close to aligned
    quant_far     = torch.zeros(1, 32) * 0.9  # close to base

    loss_close = _apc_loss(quant_close, aligned, base)
    loss_far   = _apc_loss(quant_far,   aligned, base)
    assert loss_close.item() < loss_far.item()


def test_apc_loss_scalar():
    a = torch.randn(2, 16)
    b = torch.randn(2, 16)
    c = torch.randn(2, 16)
    loss = _apc_loss(a, b, c)
    assert loss.ndim == 0   # scalar


def test_aaq_patch_apply_dict_warns(tiny_model, caplog):
    import logging
    patch = AAQPatch(params={})
    with caplog.at_level(logging.WARNING):
        result = patch.apply({"weight": torch.zeros(8, 8)})
    assert result is not None
    assert any("apply_to_model" in r.message for r in caplog.records)


def test_aaq_patch_apply_to_model_runs(tiny_model):
    """apply_to_model should complete without error even with no ref models."""
    patch = AAQPatch(params={
        "aligned_model_path": "",    # empty → probe-free mode
        "base_model_path": "",
        "apc_weight": 0.01,
        "calibration_steps": 2,
        "lr": 1e-3,
    })
    original_weight = tiny_model.weight.data.clone()
    patch.apply_to_model(tiny_model)
    # Probe-free mode may or may not change weights (depends on torch grad flow)
    # Just assert no exception was raised and model is still usable
    assert tiny_model.weight.shape == original_weight.shape


def test_aaq_patch_revert_restores_weights(tiny_model):
    patch = AAQPatch(params={"calibration_steps": 2, "lr": 1e-3})
    original = tiny_model.weight.data.clone()
    patch.apply_to_model(tiny_model)
    patch.revert_model(tiny_model)
    assert torch.allclose(tiny_model.weight.data, original, atol=1e-5)
