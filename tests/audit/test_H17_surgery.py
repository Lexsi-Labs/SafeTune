"""Audit H17 (penalty): Surgery's regulariser must be the paper's Eq. 2-4.

Liu et al. 2026 (arXiv:2602.05228) Eq. 2-4: sink value alpha_h(X) = mean over query tokens i != k of
A[h, k, i] (k = sink token), regulariser lambda * (1/|H|) * sum_h ReLU(alpha_h(X_m) - alpha_h(X_r)).
Official Lslland/Surgery @ef7aa9a `trainer.py:78-117`: `head_attn[:, 1:, SINK]`, loss_sink = mean over
heads of ReLU(...). The old code summed over heads (|H| times larger), averaged over every query row
including the sink row 0, and included padded query rows.
"""
import pytest
import torch

from safetune.core.optim.surgery import SurgeryConfig, SurgeryWrapper


def _attn(heads=4, layers=2, T=6, seed=0, batch=1):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(batch, heads, T, T, generator=g)
    a = a / a.sum(-1, keepdim=True)
    return tuple(a.clone() for _ in range(layers))


def test_penalty_is_the_mean_over_heads_of_relu_divergence_excluding_the_sink_row():
    harm, ref = _attn(seed=0), _attn(seed=1)
    w = SurgeryWrapper(SurgeryConfig(sink_lambda=1.0))
    w.set_reference(ref)
    got = w.compute_sink_penalty(harm).item()
    d = torch.stack([(h[0, :, 1:, 0].mean(-1) - r[0, :, 1:, 0].mean(-1)) for h, r in zip(harm, ref)])
    want = torch.relu(d).mean().item()
    assert want == pytest.approx(0.0310, abs=5e-4)                 # the toy number quoted in the audit
    assert got == pytest.approx(want, rel=1e-5), f"{got} vs paper formula {want}"


def test_padded_query_rows_are_excluded():
    T = 6
    harm, ref = _attn(seed=0, batch=2, T=T), _attn(seed=1, batch=2, T=T)
    mask = torch.ones(2, T)
    mask[1, 4:] = 0                                                # second sequence has 2 pad rows
    # poison the padded rows: they must not influence the result
    harm_poisoned = tuple(h.clone() for h in harm)
    for h in harm_poisoned:
        h[1, :, 4:, 0] = 1.0
    w = SurgeryWrapper(SurgeryConfig(sink_lambda=1.0))
    w.set_reference(ref, mask)
    a = w.compute_sink_penalty(harm, mask).item()
    b = w.compute_sink_penalty(harm_poisoned, mask).item()
    assert a == pytest.approx(b, rel=1e-6)
    # and equals the formula over the valid rows only
    def alpha(layers):
        out = []
        for lyr in layers:
            num = lyr[0, :, 1:, 0].sum(-1) + lyr[1, :, 1:4, 0].sum(-1)
            out.append(num / ((T - 1) + 3))
        return torch.stack(out)
    want = torch.relu(alpha(harm) - alpha(ref)).mean().item()
    assert a == pytest.approx(want, rel=1e-5)


def test_default_sink_lambda_is_the_papers_300_everywhere():
    """Paper Table 8 / official `surgery_poison_ratio.sh:16` use 300 with the mean over heads. The old default 0.01
    gave penalty * lambda ~ 2e-4 against a cross-entropy of ~10.4 on the audit toy (ratio ~2e-5)."""
    import inspect

    import safetune.harden as H
    from safetune.runner.harden import SurgeryTrainer

    assert SurgeryConfig().sink_lambda == 300.0
    assert H.SurgeryConfig(output_dir="/tmp/h17_cfg").sink_lambda == 300.0
    assert inspect.signature(SurgeryTrainer.__init__).parameters["sink_lambda"].default == 300.0
    import re

    from safetune.runner import smoke_test
    smoke = re.search(r'"SurgeryTrainer",\s*\{"sink_lambda":\s*([0-9.e+-]+)\}', inspect.getsource(smoke_test._test_harden))
    assert smoke and float(smoke.group(1)) == 300.0, "runner/smoke_test.py still runs Surgery at the old lambda"


def test_left_padding_moves_the_sink_to_each_rows_first_token():
    """With left padding key 0 is a pad token (attention weight 0), so the penalty was exactly 0."""
    from safetune.core.optim.surgery import sink_values

    T, n_pad = 6, 2
    right = _attn(seed=0, batch=2, T=T)[0]
    right_mask = torch.ones(2, T)
    right_mask[1, T - n_pad:] = 0
    left = right.clone()
    left[1] = right[1].roll(n_pad, dims=1).roll(n_pad, dims=2)       # the same row, pads moved to the front
    left_mask = right_mask.clone()
    left_mask[1] = right_mask[1].roll(n_pad)
    assert torch.allclose(sink_values(left, left_mask), sink_values(right, right_mask), atol=1e-6)
