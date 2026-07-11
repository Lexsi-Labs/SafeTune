"""Tests for PyTorch model-mode patch application (apply_to_model)."""
import pytest

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")


def _make_linear(in_f=4, out_f=4, fill_value=0.5):
    """Create a simple nn.Linear with all weights set to fill_value."""
    model = nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        model.weight.fill_(fill_value)
    return model


# ────────────────────────────────────────────────────────────────────────────
# AntidotePatch
# ────────────────────────────────────────────────────────────────────────────

def test_antidote_apply_to_model_zeros_small_weights():
    from safetune.core.patches.antidote import AntidotePatch

    model = _make_linear(fill_value=0.05)  
    # Make some weights specifically larger/smaller so they have unique WANDA scores
    with torch.no_grad():
        model.weight.data[0, 0] = 0.01  # Smallest
        model.weight.data[0, 1] = 0.1   # Larger
    patch = AntidotePatch(prune_fraction=0.5, target_modules=[""])   
    patch.apply_to_model(model)
    zero_count = (model.weight.data == 0.0).sum().item()
    # At least some weights should be pruned
    assert zero_count > 0, f"Expected some weights to be pruned, got {zero_count} zeros."


def test_antidote_revert_model():
    from safetune.core.patches.antidote import AntidotePatch

    model = _make_linear(fill_value=0.05)
    with torch.no_grad():
        model.weight.data[0, 0] = 0.01
        model.weight.data[0, 1] = 0.1
    original_data = model.weight.data.clone()
    patch = AntidotePatch(prune_fraction=0.5, target_modules=[""])
    patch.apply_to_model(model)
    
    # Pruned state
    zero_count = (model.weight.data == 0.0).sum().item()
    assert zero_count > 0
    
    # Reverted
    patch.revert_model(model)
    assert torch.allclose(model.weight.data, original_data), "revert_model should restore original weights"


# ────────────────────────────────────────────────────────────────────────────
# MSCPProjectionPatch
# ────────────────────────────────────────────────────────────────────────────

def test_mscp_subtract_mode_apply_to_model():
    from safetune.core.patches.mscp_projection import MSCPProjectionPatch

    model = _make_linear(in_f=2, out_f=2, fill_value=1.0)  # 2x2 weight; all 1.0
    # Direction: subtract [0.1, 0.0] from each param element (broadcast).
    patch = MSCPProjectionPatch(direction=[0.1, 0.0], coefficient=1.0, mode="subtract")
    patch.apply_to_model(model)
    flat = model.weight.data.view(-1)
    # Elements aligned with direction index 0 should be reduced by 0.1.
    assert flat[0].item() == pytest.approx(0.9, abs=1e-5)


def test_mscp_revert_model():
    from safetune.core.patches.mscp_projection import MSCPProjectionPatch

    model = _make_linear(in_f=2, out_f=2, fill_value=1.0)
    original = model.weight.data.clone()
    patch = MSCPProjectionPatch(direction=[0.5, 0.5], coefficient=1.0, mode="subtract")
    patch.apply_to_model(model)
    patch.revert_model(model)
    assert torch.allclose(model.weight.data, original)


# ────────────────────────────────────────────────────────────────────────────
# NLSRPatch
# ────────────────────────────────────────────────────────────────────────────

def test_nlsr_apply_to_model_donor_map():
    from safetune.core.patches.nlsr_patch import NLSRPatch

    model = _make_linear(in_f=2, out_f=2, fill_value=0.0)
    # donor_map: for weight param, transplant index 0 with value 1.0 (blend=1.0)
    patch = NLSRPatch(
        donor_map={"weight": {0: 1.0}},
        blend=1.0,
    )
    patch.apply_to_model(model)
    flat = model.weight.data.view(-1)
    assert flat[0].item() == pytest.approx(1.0, abs=1e-5), "Index 0 should be transplanted to 1.0"
    assert flat[1].item() == pytest.approx(0.0, abs=1e-5), "Index 1 should remain 0.0"


def test_nlsr_blend_partial():
    from safetune.core.patches.nlsr_patch import NLSRPatch

    model = _make_linear(in_f=2, out_f=1, fill_value=0.0)
    patch = NLSRPatch(donor_map={"weight": {0: 1.0}}, blend=0.5)
    patch.apply_to_model(model)
    flat = model.weight.data.view(-1)
    assert flat[0].item() == pytest.approx(0.5, abs=1e-5), "blend=0.5 should interpolate to 0.5"


# ────────────────────────────────────────────────────────────────────────────
# SafeLoRAPatch
# ────────────────────────────────────────────────────────────────────────────

def test_safe_lora_patch_state_dict_merge():
    from safetune.core.patches.safe_lora_patch import SafeLoRAPatch
    import tempfile, os

    model = _make_linear(in_f=2, out_f=2, fill_value=0.0)

    # Create an "aligned" state dict with all weights = 1.0.
    aligned_sd = {"weight": torch.ones(2, 2)}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(aligned_sd, f.name)
        aligned_path = f.name

    try:
        patch = SafeLoRAPatch(aligned_state_dict_path=aligned_path, alpha=1.0)
        patch.apply_to_model(model)
        # base=0.0, aligned=1.0, alpha=1.0 → result should be 1.0
        assert torch.allclose(model.weight.data, torch.ones(2, 2))
    finally:
        os.unlink(aligned_path)


def test_safe_lora_apply_to_model_revert():
    from safetune.core.patches.safe_lora_patch import SafeLoRAPatch
    import tempfile, os

    model = _make_linear(in_f=2, out_f=2, fill_value=0.5)
    original = model.weight.data.clone()

    aligned_sd = {"weight": torch.ones(2, 2)}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(aligned_sd, f.name)
        aligned_path = f.name

    try:
        patch = SafeLoRAPatch(aligned_state_dict_path=aligned_path, alpha=0.5)
        patch.apply_to_model(model)
        patch.revert_model(model)
        assert torch.allclose(model.weight.data, original)
    finally:
        os.unlink(aligned_path)
