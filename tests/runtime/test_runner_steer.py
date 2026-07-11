"""Tests for the steer runner module."""
import pytest

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestRunnerSteer:
    def test_import_module(self):
        from safetune.runner import steer
        assert steer is not None

    def test_import_trainers(self):
        from safetune.runner.steer import (
            RefusalDirectionTrainer,
            CAATrainer,
            LinearProbeGuardTrainer,
            SCANSTrainer,
            STATrainer,
            CircuitBreakerRRTrainer,
            CASTTrainer,
            AdaSteerTrainer,
            SafeSwitchTrainer,
            CircuitBreakerTrainer,
            RepBendTrainer,
            TARSteerTrainer,
            RRFAEnsembleTrainer,
            SafeSteerTrainer,
            AlphaSteerTrainer,
        )
        for cls in [
            RefusalDirectionTrainer,
            CAATrainer,
            LinearProbeGuardTrainer,
            SCANSTrainer,
            STATrainer,
            CircuitBreakerRRTrainer,
            CASTTrainer,
            AdaSteerTrainer,
            SafeSwitchTrainer,
            CircuitBreakerTrainer,
            RepBendTrainer,
            TARSteerTrainer,
            RRFAEnsembleTrainer,
            SafeSteerTrainer,
            AlphaSteerTrainer,
        ]:
            assert cls is not None

    def test_import_decoding_trainers(self):
        from safetune.runner.steer import (
            SafeDecodingTrainer,
            ContrastiveDecodingTrainer,
            ProxyTuningTrainer,
            NudgingTrainer,
        )
        for cls in [
            SafeDecodingTrainer,
            ContrastiveDecodingTrainer,
            ProxyTuningTrainer,
            NudgingTrainer,
        ]:
            assert cls is not None

    def test_load_steer_data_exists(self):
        from safetune.runner.steer import load_steer_data
        assert callable(load_steer_data)
