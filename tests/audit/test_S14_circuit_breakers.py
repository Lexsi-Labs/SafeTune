"""Audit S14: Circuit Breakers (GraySwanAI/circuit-breakers@a1ba1a5, src/lorra_circuit_breaker.py:68-76,103-104,115-118).

Official: the retain term stacks ALL hidden states; the cosine (rerouting) term uses the target layers only. The wrapper used the
target layers for both. (The padded-position denominator of the official retain term is deliberately not mirrored: ours divides by
real tokens.)
"""
import pytest
import torch

import safetune.steer as S


def _hidden(layers, B=3, T=5, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {l: torch.randn(B, T, D, generator=g) for l in layers}


def _cb(**kw):
    return S.CircuitBreakerModel(torch.nn.Linear(2, 2), target_layers=[2, 4], rr_alpha=5.0, **kw)


def test_the_retain_term_covers_every_layer_given_not_only_the_target_layers():
    all_layers = [0, 1, 2, 3, 4]
    orr, lr_ = _hidden(all_layers, seed=1), _hidden(all_layers, seed=2)
    oc, lc = _hidden([2, 4], seed=3), _hidden([2, 4], seed=4)
    got = _cb().compute_rr_loss(orr, lr_, oc, lc, step=300)           # step 300: only the retain term (coefficient alpha)
    want = 5.0 * torch.stack([(lr_[l] - orr[l]).norm(dim=-1).mean() for l in all_layers]).mean()
    assert torch.allclose(got, want, atol=1e-5), f"{float(got):.4f} vs {float(want):.4f}"


def test_retain_dicts_with_only_the_target_layers_are_unchanged():
    orr, lr_ = _hidden([2, 4], seed=1), _hidden([2, 4], seed=2)
    oc, lc = _hidden([2, 4], seed=3), _hidden([2, 4], seed=4)
    mask = torch.ones(3, 5)
    mask[1, 3:] = 0
    got = _cb().compute_rr_loss(orr, lr_, oc, lc, step=150, retain_mask=mask, cb_mask=mask)
    ref = S.CircuitBreakerRRModel(torch.nn.Linear(2, 2), config=S.CircuitBreakerRRConfig(alpha=5.0))
    want = ref.rr_loss(lc, oc, lr_, orr, step=150, cb_mask=mask, retain_mask=mask)
    assert torch.allclose(got, want, atol=1e-6)


def test_runtime_hook_raises_when_no_hook_would_be_installed():
    m = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="runtime_hook"):
        S.CircuitBreakerModel(m, unsafe_directions={2: torch.randn(8)}, runtime_hook=True)       # layer 2 is not a target layer
    with pytest.raises(ValueError, match="runtime_hook"):
        S.CircuitBreakerModel(m, runtime_hook=True)                                              # no directions at all


def test_runtime_hook_still_installs_when_a_direction_layer_is_hookable():
    from _steer_common import build
    m, _ = build()
    cb = S.CircuitBreakerModel(m, unsafe_directions={2: torch.randn(64)}, target_layers=[2, 3], runtime_hook=True)
    assert sum(len(l._forward_hooks) for l in m.model.layers) == 2
    cb.remove_hooks()
    assert sum(len(l._forward_hooks) for l in m.model.layers) == 0
