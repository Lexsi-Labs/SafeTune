"""Unlearn trainers train in fp32 and refuse to return a model with NaN weights."""
import pytest
import torch
from torch import nn

from safetune.runner.unlearn._base import _UnlearnBase


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(4, 4)


class _Ok(_UnlearnBase):
    METHOD = "Ok"

    def unlearn(self, forget, retain):
        return self.model


class _Breaks(_UnlearnBase):
    METHOD = "Breaks"

    def unlearn(self, forget, retain):
        with torch.no_grad():
            self.model.w.weight.fill_(float("nan"))
        return self.model


def test_fp16_model_is_trained_in_fp32():
    out = _Ok(_Tiny().half(), model_id="tiny").unlearn([], [])
    assert all(p.dtype == torch.float32 for p in out.parameters())


def test_upcast_fp16_false_keeps_fp16():
    with pytest.warns(DeprecationWarning, match="upcast_fp16 is deprecated"):
        trainer = _Ok(_Tiny().half(), model_id="tiny", upcast_fp16=False)
    out = trainer.unlearn([], [])
    assert all(p.dtype == torch.float16 for p in out.parameters())


def test_bf16_model_is_trained_in_fp32():
    """Was left in bf16, where lr-1e-5 updates round away (Tiny Aya, U1)."""
    out = _Ok(_Tiny().to(torch.bfloat16), model_id="tiny").unlearn([], [])
    assert all(p.dtype == torch.float32 for p in out.parameters())


def test_upcast_false_keeps_bf16():
    out = _Ok(_Tiny().to(torch.bfloat16), model_id="tiny", upcast=False).unlearn([], [])
    assert all(p.dtype == torch.bfloat16 for p in out.parameters())


def test_nan_weights_raise():
    with pytest.raises(RuntimeError, match="Breaks: 1 weight tensor"):
        _Breaks(_Tiny(), model_id="tiny").unlearn([], [])
