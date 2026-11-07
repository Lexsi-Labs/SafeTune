"""Audit E07: SpectralEntropyMonitor (no paper: maths and edges). Random tiny Llama, placeholder strings."""
import math

import pytest
import torch

from safetune.evaluate import SpectralEntropyMonitor, SpectralMonitorConfig
from safetune.instrumentation.evaluate.suite.spectral import _spectral_entropy

from _eval_common import build

BENIGN = ["PLACEHOLDER benign request number %d about an ordinary thing" % i for i in range(6)]


def test_entropy_matches_a_direct_computation_on_a_random_matrix():
    torch.manual_seed(0)
    a = torch.randn(20, 32).double()
    c = a - a.mean(0, keepdim=True)
    s2 = torch.linalg.svdvals(c) ** 2
    p = s2 / s2.sum()
    assert abs(_spectral_entropy(a.float(), 1e-12) - float(-(p * p.log()).sum())) < 1e-4


def test_a_collapsed_input_is_not_maximum_entropy():
    """Identical rows centre to all zeros (rank 0); the old absolute floor made the spectrum uniform, H = log(n), the MAXIMUM."""
    h = _spectral_entropy(torch.ones(10, 64), 1e-12)
    assert h == 0.0, f"constant hidden states read as H={h:.3f} (max {math.log(10):.3f})"


def test_entropy_does_not_depend_on_the_activation_scale():
    torch.manual_seed(1)
    a = torch.randn(10, 64)
    ref = _spectral_entropy(a, 1e-12)
    for scale in (1e-8, 1e-4, 1.0, 1e4):
        assert abs(_spectral_entropy(a * scale, 1e-12) - ref) < 1e-4, scale


@pytest.mark.parametrize("bad", ["nan", "inf", "one_row", "empty"])
def test_unscorable_input_is_nan_not_zero_entropy(bad):
    a = torch.randn(10, 64)
    if bad == "nan":
        a[3, 5] = float("nan")
    elif bad == "inf":
        a[3, 5] = float("inf")
    elif bad == "one_row":
        a = a[:1]
    else:
        a = a[:0]
    assert math.isnan(_spectral_entropy(a, 1e-12))


def test_well_conditioned_entropy_is_unchanged_to_1e_6():
    """Already-correct path: the relative floor leaves a normal spectrum alone."""
    torch.manual_seed(2)
    for shape in ((20, 32), (100, 64), (7, 128)):
        a = torch.randn(*shape) * 3
        c = a.double() - a.double().mean(0, keepdim=True)
        s2 = torch.linalg.svdvals(c) ** 2
        p = s2 / s2.sum()
        old = float(-(p.clamp_min(1e-12 / s2.sum()) * p.clamp_min(1e-300).log()).sum())
        assert abs(_spectral_entropy(a, 1e-12) - old) < 1e-6


def test_scan_leaves_a_nan_prompt_unflagged_and_calibrate_leaves_it_out(monkeypatch):
    m, tok = build()
    mon = SpectralEntropyMonitor(m, tok)
    vals = iter([{0: 2.0}, {0: float("nan")}, {0: 2.2}, {0: 2.1}])
    monkeypatch.setattr(mon, "_run_prompts", lambda ps: [next(vals) for _ in ps])
    base = mon.calibrate(["a", "b", "c"])
    assert abs(base[0][0] - 2.1) < 1e-9                       # mean of 2.0 and 2.2 only
    assert mon.scan(["d"]) == []


def test_an_out_of_range_target_layer_raises_instead_of_monitoring_nothing():
    m, tok = build()
    mon = SpectralEntropyMonitor(m, tok, SpectralMonitorConfig(target_layers=[99]))
    with pytest.raises(ValueError, match="target_layers"):
        mon.calibrate(BENIGN)


def test_calibrating_on_one_prompt_does_not_produce_a_1e_6_std():
    """With a single prompt the std is set to 1e-6, so any later prompt is thousands of sigma away."""
    m, tok = build()
    mon = SpectralEntropyMonitor(m, tok)
    with pytest.raises(ValueError, match="at least 2"):
        mon.calibrate(BENIGN[:1])


def test_scan_and_trajectory_are_in_input_order_and_batch_invariant():
    """Already-correct path: results do not depend on the batch size or the padding."""
    m, tok = build()
    ps = [BENIGN[0], BENIGN[1] + " with a few more words in it", BENIGN[2]]
    a = SpectralEntropyMonitor(m, tok, SpectralMonitorConfig(batch_size=1))._run_prompts(ps)
    b = SpectralEntropyMonitor(m, tok, SpectralMonitorConfig(batch_size=3))._run_prompts(ps)
    for x, y in zip(a, b):
        assert x.keys() == y.keys() and all(abs(x[k] - y[k]) < 1e-3 for k in x)
