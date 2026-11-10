"""RESTA with DARE drops the fine-tuning delta, not the safety vector.

Bhardwaj et al. (arXiv:2402.11746, section 2, Eq. 7-9 and section 4): "we perform Drop and REscale
operations (DARE) on the delta parameters ... DARE applies drop and rescale to the delta parameters
of the SFT model", i.e. on ``theta_SFT - theta_aligned``; the safety vector is then added whole:

    theta = theta_aligned + DARE(theta_SFT - theta_aligned, p) + gamma * (theta_aligned - theta_unaligned)

apply_resta(dare=True) dropped entries of the safety vector instead and left the fine-tuning delta
alone, so "RESTA_d" here was a different method from the paper's.
"""
import copy

import torch

from safetune.recover import apply_resta


class _M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(48, 48, bias=False)


def _model(seed):
    torch.manual_seed(seed)
    return _M()


def test_dare_leaves_a_model_with_no_finetuning_delta_alone():
    """ft == aligned: there is nothing to drop, so DARE must not change the result."""
    base, aligned = _model(0), _model(1)
    plain = apply_resta(copy.deepcopy(aligned), base=base, aligned=aligned, alpha=0.5)
    dare = apply_resta(copy.deepcopy(aligned), base=base, aligned=aligned, alpha=0.5,
                       dare=True, dare_drop_rate=0.5, dare_seed=0)
    assert torch.allclose(dare.q_proj.weight, plain.q_proj.weight, atol=1e-6)


def test_dare_drops_and_rescales_the_finetuning_delta_and_adds_the_whole_safety_vector():
    base, aligned, ft = _model(0), _model(1), _model(2)
    p, alpha = 0.5, 0.7
    out = apply_resta(copy.deepcopy(ft), base=base, aligned=aligned, alpha=alpha,
                      dare=True, dare_drop_rate=p, dare_seed=3).q_proj.weight.detach()

    a, b, f = aligned.q_proj.weight.detach(), base.q_proj.weight.detach(), ft.q_proj.weight.detach()
    kept = out - a - alpha * (a - b)                      # what is left of the fine-tuning delta
    sft = f - a
    dropped = kept.abs() < 1e-6
    assert 0.35 < dropped.float().mean().item() < 0.65    # about p of its entries are dropped
    assert torch.allclose(kept[~dropped], sft[~dropped] / (1 - p), atol=1e-5)   # the rest rescaled by 1/(1-p)


def test_dare_seed_makes_the_mask_reproducible():
    """The same dare_seed gives the same result; another seed drops other entries."""
    base, aligned, ft = _model(0), _model(1), _model(2)

    def run(seed):
        return apply_resta(copy.deepcopy(ft), base=base, aligned=aligned, alpha=0.7,
                           dare=True, dare_drop_rate=0.5, dare_seed=seed).q_proj.weight.detach()

    assert torch.equal(run(3), run(3))
    assert not torch.equal(run(3), run(4))
