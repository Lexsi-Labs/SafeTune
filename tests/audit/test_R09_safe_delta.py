"""Audit R09: without a Hessian, Safe Delta's keep-set depends on argsort tie order, not on |delta|.

With H = I every entry has importance 1/d^2 == 1, so argsort(importance) is all ties and the
greedy budget walk admits an arbitrary prefix of entries. The docstring says this "reduces to the
magnitude of the delta". Reordering the rows of the delta must not change which delta values are
kept if the rule depends only on their magnitudes.
"""
import torch

from safetune.interventions.recover.safe_delta import _adjust_delta


def _kept_values(delta: torch.Tensor, s: float = 5.0) -> torch.Tensor:
    w_align = torch.zeros_like(delta)
    out = _adjust_delta(W_align=w_align, W_sft=delta, s=s, hessian=None, importance_drop=0.15,
                        percdamp=0.01, ref_columns=4096, blocksize=2048, sub_block_size=4)
    return torch.sort(out[out != 0].abs()).values


def test_safe_delta_data_free_selection_depends_only_on_delta_magnitude():
    torch.manual_seed(0)
    delta = torch.randn(64, 64) * torch.rand(64, 1) * 0.1      # rows with different scales
    perm = torch.randperm(64)
    a, b = _kept_values(delta), _kept_values(delta[perm])
    assert a.numel() == b.numel() and torch.allclose(a, b), (
        f"kept {a.numel()} vs {b.numel()} entries; mean |delta| kept {a.mean():.4f} vs {b.mean():.4f}"
    )
