"""SafeMERGE's dare_linear drops entries of both deltas, as the authors' PEFT-based code does.

The authors merge through PEFT's ``dare_linear``, which prunes and rescales every task tensor it is
given (the fine-tuned update and the safe one) before the weighted sum. apply_safemerge used to drop
only the safe delta.
"""
import copy

import torch

from safetune.interventions.recover import safemerge
from safetune.recover import apply_safemerge


class _M(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.q_proj = torch.nn.Linear(40, 40, bias=False)


def _merge(**kw):
    base, aligned, ft = _M(0), _M(1), _M(2)
    out = apply_safemerge(copy.deepcopy(ft), base=base, aligned=aligned, threshold=2.0, alpha=0.5, **kw)
    return out.q_proj.weight.detach(), base.q_proj.weight.detach()


def test_both_deltas_go_through_dare(monkeypatch):
    calls = []
    real = safemerge._dare_drop
    monkeypatch.setattr(safemerge, "_dare_drop", lambda d, density: calls.append(d.clone()) or real(d, density))
    _merge(merge_type="dare_linear", density=0.5)
    assert len(calls) == 2                       # the safe delta and the fine-tuned delta, once each


def test_density_one_equals_linear():
    dare, _ = _merge(merge_type="dare_linear", density=1.0)
    lin, _ = _merge(merge_type="linear")
    assert torch.allclose(dare, lin, atol=1e-6)


def test_both_deltas_are_thinned_and_rescaled():
    torch.manual_seed(7)
    p = 0.5
    out, base = _merge(merge_type="dare_linear", density=p)
    lin, _ = _merge(merge_type="linear")
    kept_share = ((out - base).abs() > 1e-9).float().mean().item()
    # each delta keeps about `density` of its entries, so the sum is non-zero where either kept one
    assert 0.6 < kept_share < 0.9
    assert not torch.allclose(out, lin)
