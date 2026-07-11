"""Regression tests for the two train/unlearn execution bugs found by running
every method end-to-end on a real model:

1. SimDPOTrainer.unlearn required pre-built {chosen, rejected} pairs and crashed
   on raw forget batches; now it auto-builds pairs (like FLATTrainer).
2. SaLoRATrainer.train built safety_inputs with torch.tensor([ex["input_ids"]]),
   which raised on torch-formatted datasets; now uses torch.as_tensor(...).view.
"""
import inspect

import pytest

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


class TestSimDPOAutoBuildsPairs:
    def _trainer(self, monkeypatch):
        import torch.nn as nn
        import safetune.unlearn as SU
        import safetune.runner.utils.model_utils as MU
        from safetune.runner.unlearn import SimDPOTrainer
        t = SimDPOTrainer(nn.Linear(2, 2), model_id="dummy")
        monkeypatch.setattr(t, "_wrap_lora", lambda m: m)
        monkeypatch.setattr(t, "_maybe_merge", lambda m: m)
        monkeypatch.setattr(t, "_to_device", lambda b: b)
        monkeypatch.setattr(SU, "simdpo_unlearn", lambda model, **k: model)
        monkeypatch.setattr(MU, "load_tok", lambda mid: object())
        return t

    def test_raw_forget_triggers_pair_build(self, monkeypatch):
        t = self._trainer(monkeypatch)
        built = {}
        monkeypatch.setattr(t, "make_simdpo_pairs",
                            lambda forget, tok: built.setdefault("n", len(list(forget)))
                            or [{"chosen": {}, "rejected": {}}])
        t.unlearn([{"input_ids": [1, 2, 3]}], [])   # raw -> must auto-build
        assert built.get("n") == 1

    def test_prebuilt_pairs_are_not_rebuilt(self, monkeypatch):
        t = self._trainer(monkeypatch)
        monkeypatch.setattr(t, "make_simdpo_pairs",
                            lambda forget, tok: (_ for _ in ()).throw(
                                AssertionError("should not rebuild pre-built pairs")))
        t.unlearn([{"chosen": {}, "rejected": {}}], [])  # already pairs -> pass through


class TestSaLoRASafetyInputsRobust:
    def test_train_no_longer_wraps_tensor_in_list(self):
        from safetune.runner.harden import SaLoRATrainer
        src = inspect.getsource(SaLoRATrainer.train)
        assert 'torch.tensor([ex["input_ids"]]' not in src
        assert "torch.as_tensor(ex[\"input_ids\"]" in src

    def test_as_tensor_view_handles_both_list_and_tensor(self):
        import torch
        for ids in ([1, 2, 3], torch.tensor([1, 2, 3])):
            out = torch.as_tensor(ids, dtype=torch.long).view(1, -1)
            assert out.shape == (1, 3)
