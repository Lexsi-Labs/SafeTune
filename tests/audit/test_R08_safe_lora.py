"""Audit R08: Safe LoRA's state-dict path projects ft - base, not the drift ft - aligned.

Hsu et al. 2024 / IBM/SafeLoRA model.py: the projection acts on the LoRA update (the drift
from the aligned model the adapter was trained on): W_new = W_aligned + P (W_ft - W_aligned).
A model with no drift (ft == aligned) must be left unchanged. apply_safe_lora(threshold=1.0)
projects every layer; on the state-dict path it returns base + P (aligned - base) instead.
"""
import copy

import torch

from safetune.recover import apply_safe_lora


def _lin(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(32, 32, bias=False))


def test_safe_lora_leaves_an_undrifted_aligned_model_unchanged():
    base, aligned = _lin(0), _lin(1)
    ft = copy.deepcopy(aligned)                        # no drift at all
    before = ft[0].weight.detach().clone()

    apply_safe_lora(ft, aligned_state_dict=aligned.state_dict(),
                    base_state_dict=base.state_dict(), threshold=1.0, target_modules=["0"])

    assert torch.allclose(ft[0].weight, before, atol=1e-5), (
        f"max |diff| = {(ft[0].weight - before).abs().max().item():.3f}"
    )
