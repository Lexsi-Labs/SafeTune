"""
Tests for Runtime Inference features (Dynamic Patching & Safeguard).
"""

import pytest
try:
    import torch
    import torch.nn as nn
except ImportError:
    pytest.skip("PyTorch not available", allow_module_level=True)

from safetune.core.runtime.inference import (
    DynamicPatchingConfig,
    HookedGenerationWrapper,
    LLMSafeguardPredictor,
    PatchingStrategy,
)


class DummyModel(nn.Module):
    """A trivial generic sequence model."""
    def __init__(self):
        super().__init__()
        # A single layer just mapping input to output structurally
        self.layer = nn.Linear(4, 4)

    def forward(self, input_ids=None, **kwargs):
        # Dummy pass: expect input_ids to be [B, S, D]
        if input_ids is None:
            input_ids = torch.ones(1, 1, 4)
        out = self.layer(input_ids)
        return out

    def generate(self, *args, **kwargs):
        # Fake generate loop: just calls forward
        return self.forward(*args, **kwargs)


def test_dynamic_patching():
    base = DummyModel()
    guided = DummyModel()
    
    # We want guided model to output something different so we verify the patch
    with torch.no_grad():
        guided.layer.weight.fill_(2.0)
        guided.layer.bias.fill_(1.0)
        base.layer.weight.fill_(0.0)
        base.layer.bias.fill_(0.0)

    config = DynamicPatchingConfig(
        target_modules=["layer"],
        strategy=PatchingStrategy(mode="replace", scale=1.0)
    )

    wrapper = HookedGenerationWrapper(base_model=base, guided_model=guided, config=config)
    
    inputs = torch.ones(1, 1, 4)
    out_vanilla = base(inputs)
    assert out_vanilla.sum().item() == 0.0  # base outputs zeros
    
    out_patched = wrapper.generate(input_ids=inputs)
    # The patch replaces base.layer output with guided.layer output
    # Guided linear is (ones * 2.0 * 4_features) + 1.0 = 9.0 per feature
    assert out_patched.sum().item() > 0.0
    assert torch.allclose(out_patched, torch.ones(1, 1, 4) * 9.0)


class DummyClassifier:
    """Fake classifier predicting based on a static feature value."""
    def predict_proba(self, X):
        import numpy as np
        # Let's say it returns unsafe (col 1) = 1.0 if it sees the number 9.0
        # X shape: (batch, features)
        unsafe_prob = 1.0 if X.sum() >= 9.0 else 0.0
        return np.array([[1.0 - unsafe_prob, unsafe_prob]])


def test_llm_safeguard():
    base = DummyModel()
    with torch.no_grad():
        base.layer.weight.fill_(0.0)
        base.layer.bias.fill_(5.0)  # output is 5.0 for all

    classifier = DummyClassifier()
    # Let's map module "layer" -> indices [0, 1]
    safeguard = LLMSafeguardPredictor(
        model=base,
        classifier=classifier,
        target_indices={"layer": [0, 1]},
        threshold=0.5,
    )

    inputs = torch.ones(1, 1, 4)
    # 5.0 + 5.0 = 10.0 > 9.0 -> predict_proba gives 1.0 -> intercepted!
    out = safeguard.generate_safe(input_ids=inputs)
    
    # safeguard should return the inputs unaltered before generation completes
    assert torch.equal(out, inputs)
    assert safeguard.is_currently_unsafe is True
