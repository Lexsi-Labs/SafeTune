"""Tests for the harden runner module."""
import pytest

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestRunnerHarden:
    def test_import_module(self):
        from safetune.runner import harden
        assert harden is not None

    def test_import_trainers(self):
        from safetune.runner.harden import (
            PlainSFTTrainer,
            SafeGradTrainer,
            LisaTrainer,
            SPPFTTrainer,
            LookAheadTrainer,
            STARDSSTrainer,
            DeRTaTrainer,
            AsFTTrainer,
            SAPTrainer,
            SurgeryTrainer,
            BoosterTrainer,
            VaccineTrainer,
            TVaccineTrainer,
            RepNoiseTrainer,
            CTRAPTrainer,
            SEAMTrainer,
            DOORTrainer,
            TARTrainer,
            SaLoRATrainer,
            SEALTrainer,
            ConstrainedSFTTrainer,
            LoXHardenTrainer,
        )
        for cls in [
            PlainSFTTrainer,
            SafeGradTrainer,
            LisaTrainer,
            SPPFTTrainer,
            LookAheadTrainer,
            STARDSSTrainer,
            DeRTaTrainer,
            AsFTTrainer,
            SAPTrainer,
            SurgeryTrainer,
            BoosterTrainer,
            VaccineTrainer,
            TVaccineTrainer,
            RepNoiseTrainer,
            CTRAPTrainer,
            SEAMTrainer,
            DOORTrainer,
            TARTrainer,
            SaLoRATrainer,
            SEALTrainer,
            ConstrainedSFTTrainer,
            LoXHardenTrainer,
        ]:
            assert cls is not None

    def test_load_harden_data_exists(self):
        from safetune.runner.harden import load_harden_data
        assert callable(load_harden_data)
