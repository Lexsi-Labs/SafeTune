"""Tests for the recover runner module."""
import pytest

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestRunnerRecover:
    def test_import_module(self):
        from safetune.runner import recover
        assert recover is not None

    def test_import_trainers(self):
        from safetune.runner.recover import (
            CThetaTrainer,
            TaskArithmeticTrainer,
            PrePostMergeTrainer,
            SOMFTrainer,
            WiseFTTrainer,
            ReStaTrainer,
            SafeMergeTrainer,
            SafeDeltaTrainer,
            SafeLoRATrainer,
            QReSafeTrainer,
            AAQTrainer,
            RepNoiseRecoverTrainer,
            LoXTrainer,
            LSSFTrainer,
            SafetyVectorRestoreTrainer,
            NLSRTrainer,
            AntidoteTrainer,
            AntidoteV2Trainer,
            MSCPTrainer,
            GradSelectiveRecoverTrainer,
            OneShotSafetyPatchTrainer,
            PKETrainer,
            SafeReActTrainer,
            SCRUBTrainer,
        )
        for cls in [
            CThetaTrainer,
            TaskArithmeticTrainer,
            PrePostMergeTrainer,
            SOMFTrainer,
            WiseFTTrainer,
            ReStaTrainer,
            SafeMergeTrainer,
            SafeDeltaTrainer,
            SafeLoRATrainer,
            QReSafeTrainer,
            AAQTrainer,
            RepNoiseRecoverTrainer,
            LoXTrainer,
            LSSFTrainer,
            SafetyVectorRestoreTrainer,
            NLSRTrainer,
            AntidoteTrainer,
            AntidoteV2Trainer,
            MSCPTrainer,
            GradSelectiveRecoverTrainer,
            OneShotSafetyPatchTrainer,
            PKETrainer,
            SafeReActTrainer,
            SCRUBTrainer,
        ]:
            assert cls is not None

    def test_load_recover_data_exists(self):
        from safetune.runner.recover import load_recover_data
        assert callable(load_recover_data)
