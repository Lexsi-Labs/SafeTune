"""Audit R04: SOMF task vectors are taken relative to the aligned model, not the base.

Yi et al. 2024 (arXiv:2405.09055) Algorithm 1 line 1: delta_i = theta_i - theta_SAFE, and the
fused model is theta_SAFE + masked delta (official xinykou/safety_realignment
FTMaskModel._init_task_vector: state_dict_sub(finetuned, model_name_or_path)).
With an all-ones mask the fused model is therefore the fine-tuned model itself.
"""
import copy

import torch

from safetune.recover import somf_merge


def _lin(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False))


def test_somf_all_ones_mask_returns_the_finetuned_model():
    base, aligned = _lin(0), _lin(1)
    ft = copy.deepcopy(aligned)                       # SFT started from the aligned model
    with torch.no_grad():
        ft[0].weight.add_(0.1 * torch.randn(16, 16))
    expected = ft[0].weight.detach().clone()          # theta_SAFE + 1 * (theta_ft - theta_SAFE)
    ones = {"0.weight": torch.ones(16, 16, dtype=torch.bool)}

    somf_merge(ft, base=base, aligned=aligned, lam=1.0, subspace_mask=ones)

    assert torch.allclose(ft[0].weight, expected, atol=1e-5), (
        f"max |diff| = {(ft[0].weight - expected).abs().max().item():.3f}: "
        "the aligned-minus-base safety vector was added on top"
    )
