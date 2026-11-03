"""Audit R08 (review of #109): a missing base must never fall back to the model's own weights.

With no base_state_dict on a non-PEFT model the base used to be the model's own weights, so the
delta became ft - aligned and the weights moved silently (main left the model untouched).
"""
import copy

import pytest
import torch

from safetune.recover import apply_safe_lora
from safetune.runner import recover


class _M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(16, 16, bias=False)


def _models():
    torch.manual_seed(0)
    base = _M()
    aligned = copy.deepcopy(base)
    ft = copy.deepcopy(base)
    with torch.no_grad():
        aligned.q_proj.weight.add_(0.3 * torch.randn(16, 16))
        ft.q_proj.weight.copy_(aligned.q_proj.weight + 0.1 * torch.randn(16, 16))
    return base, aligned, ft


def test_missing_base_raises_instead_of_moving_weights():
    _, aligned, ft = _models()
    before = ft.q_proj.weight.detach().clone()
    with pytest.raises(ValueError, match="base_state_dict"):
        apply_safe_lora(ft, aligned_state_dict=aligned.state_dict(), threshold=0.99)
    assert torch.equal(ft.q_proj.weight, before)


def test_runner_takes_the_base_from_base_model():
    base, aligned, ft = _models()
    ref = copy.deepcopy(ft)
    apply_safe_lora(ref, aligned_state_dict=aligned.state_dict(),
                    base_state_dict=base.state_dict(), threshold=0.99)
    before = ft.q_proj.weight.detach().clone()
    recover.SafeLoRATrainer(ft, base_model=base, aligned_model=aligned, threshold=0.99).apply()
    assert not torch.equal(ft.q_proj.weight, before)       # it projected
    assert torch.equal(ft.q_proj.weight, ref.q_proj.weight)
