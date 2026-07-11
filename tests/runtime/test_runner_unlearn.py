"""Tests for the unlearn runner module."""
import pytest

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestRunnerUnlearn:
    def test_import_module(self):
        from safetune.runner import unlearn
        assert unlearn is not None

    def test_import_trainers(self):
        from safetune.runner.unlearn import (
            RMUTrainer,
            NPOTrainer,
            GradientAscentTrainer,
            GradDiffTrainer,
            FLATTrainer,
            SimDPOTrainer,
        )
        assert RMUTrainer is not None
        assert NPOTrainer is not None
        assert GradientAscentTrainer is not None
        assert GradDiffTrainer is not None
        assert FLATTrainer is not None
        assert SimDPOTrainer is not None

    def test_trainer_base_class(self):
        from safetune.runner.unlearn._base import _UnlearnBase
        assert _UnlearnBase is not None

    def test_load_unlearn_data_exists(self):
        from safetune.runner.unlearn import load_unlearn_data
        assert callable(load_unlearn_data)
