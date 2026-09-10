"""place_model must follow the live accelerator, not a hardcoded CUDA/dtype."""
import pytest

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None


@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestPlaceModel:
    def test_accelerator_prefers_cuda_then_mps_then_cpu(self):
        from safetune.runner.utils.model_utils import accelerator_device

        dev = accelerator_device()
        if torch.cuda.is_available():
            assert dev.type == "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            assert dev.type == "mps"
        else:
            assert dev.type == "cpu"

    def test_place_model_does_not_change_dtype(self):
        from safetune.runner.utils.model_utils import place_model, accelerator_device

        m = nn.Linear(4, 4)
        src_dtype = next(m.parameters()).dtype
        out = place_model(m)
        p = next(out.parameters())
        assert p.dtype == src_dtype
        assert p.device.type == accelerator_device().type

    def test_place_model_skips_dispatched_models(self):
        from safetune.runner.utils.model_utils import place_model

        m = nn.Linear(4, 4)
        m.hf_device_map = {"": "cpu"}
        assert place_model(m) is m
        assert next(m.parameters()).device.type == "cpu"
