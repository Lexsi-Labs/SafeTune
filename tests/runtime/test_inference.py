"""Tests for runtime inference modules: dynamic patching and safeguard."""
import pytest

try:
    import torch
except ImportError:
    torch = None  # noqa: F811


@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestDynamicPatchingConfig:
    def test_default_config(self):
        from safetune.core.runtime.inference import DynamicPatchingConfig
        cfg = DynamicPatchingConfig()
        assert cfg.target_modules is None or cfg.target_modules == []

    def test_custom_config(self):
        from safetune.core.runtime.inference import DynamicPatchingConfig
        cfg = DynamicPatchingConfig(
            target_modules=["mlp"],
            target_indices=[0, 5, 10],
        )
        assert cfg.target_modules == ["mlp"]
        assert cfg.target_indices == [0, 5, 10]


@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestPatchingStrategy:
    def test_add_strategy(self):
        from safetune.core.runtime.inference import PatchingStrategy
        s = PatchingStrategy(mode="add", scale=0.5)
        assert s.mode == "add"
        assert s.scale == 0.5

    def test_replace_strategy(self):
        from safetune.core.runtime.inference import PatchingStrategy
        s = PatchingStrategy(mode="replace", scale=1.0)
        assert s.mode == "replace"
        assert s.scale == 1.0


@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestLLMSafeguardPredictor:
    def test_import(self):
        from safetune.core.runtime.inference import LLMSafeguardPredictor
        assert LLMSafeguardPredictor is not None
