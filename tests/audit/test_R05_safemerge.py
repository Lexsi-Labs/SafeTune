"""Audit R05: SafeMERGE gates on the fine-tuning update, not on W_ft - W_base.

Djuhera et al. 2025 (arXiv:2503.17239, Sec. 2, Eq. 2): rho = cos(dW_f, C dW_f), where dW_f is the
LoRA update the task fine-tune added to the aligned model and C = V V^T / ||V||_F with
V = W_aligned - W_base. W_ft - W_base is dW_f + V, and C V points along V, so the safety vector
props rho up whatever the fine-tune did. Below, an update that lies almost outside the safety
subspace (rho = 0.10, so the layer must be merged at tau = 0.35) scored 0.71 and was left alone.
"""
import copy

import torch

from safetune.recover import apply_safemerge


def _lin() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False))


def _models(update_in_subspace: bool):
    torch.manual_seed(0)
    base = _lin()
    aligned, ft = copy.deepcopy(base), copy.deepcopy(base)
    e = torch.eye(16)
    with torch.no_grad():
        aligned[0].weight.add_(torch.outer(e[0], e[1]))             # V = u v^T, ||V||_F = 1
        # Inside the subspace: along u. Outside: mostly e[2], which is orthogonal to u.
        row = e[0] if update_in_subspace else e[2] + 0.1 * e[0]
        ft[0].weight.copy_(aligned[0].weight + torch.outer(row, e[3]))
    return base, aligned, ft


def test_update_outside_the_safety_subspace_is_merged():
    base, aligned, ft = _models(update_in_subspace=False)
    expected = 0.5 * ft[0].weight.detach() + 0.5 * aligned[0].weight.detach()

    apply_safemerge(ft, base=base, aligned=aligned, threshold=0.35, alpha=0.5)

    diff = (ft[0].weight - expected).abs().max().item()
    assert diff < 1e-6, f"layer was not merged (max |diff| from the merge = {diff:.3f})"


def test_update_inside_the_safety_subspace_is_kept():
    base, aligned, ft = _models(update_in_subspace=True)
    before = ft[0].weight.detach().clone()

    apply_safemerge(ft, base=base, aligned=aligned, threshold=0.35, alpha=0.5)

    assert torch.equal(ft[0].weight, before)
