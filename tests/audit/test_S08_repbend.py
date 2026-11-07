"""Audit S08: RepBend (AIM-Intelligence/RepBend@4f49776, methods/rep_bending/trainer.py:303,316).

The official safe / unsafe terms take the L2 norm of every token's difference and average; the wrapper averaged the tokens first
and took the norm of the mean difference (which is always smaller). The docstring and the docs do not declare that as an adaptation.
"""
import torch

import safetune.steer as S


def _loss(**kw):
    m = torch.nn.Linear(2, 2)
    return S.RepBendModel(m, target_layers=[1], **kw)


def test_safe_and_unsafe_terms_are_the_mean_of_per_token_norms():
    torch.manual_seed(0)
    B, T, D = 4, 7, 16
    h, h0 = {1: torch.randn(B, T, D)}, {1: torch.randn(B, T, D)}
    u, u0 = {1: torch.randn(B, T, D)}, {1: torch.randn(B, T, D)}
    safe = _loss(loss_alpha=1.0, loss_beta=0.0, loss_gamma=0.0, loss_epsilon=0.0).compute_repbend_loss(h, h0, u, u0)
    unsafe = _loss(loss_alpha=0.0, loss_beta=1.0, loss_gamma=0.0, loss_epsilon=0.0).compute_repbend_loss(h, h0, u, u0)
    assert torch.allclose(safe, torch.norm(h[1] - h0[1], dim=-1).mean(), atol=1e-6), float(safe)
    assert torch.allclose(unsafe, -torch.norm(u[1] - u0[1], dim=-1).mean(), atol=1e-6), float(unsafe)


def test_already_pooled_inputs_are_unchanged():
    torch.manual_seed(1)
    h, h0 = {1: torch.randn(4, 16)}, {1: torch.randn(4, 16)}
    got = _loss(loss_alpha=1.0, loss_beta=0.0, loss_gamma=0.0, loss_epsilon=0.0).compute_repbend_loss(h, h0, h, h0)
    assert torch.equal(got, (h[1] - h0[1]).norm(dim=-1).mean())


# ---- the cosine term at the request position (trainer.py:354) ---------------------------------------------------

def _cos_inputs():
    torch.manual_seed(3)
    B, T, D = 4, 7, 16
    h, h0 = {1: torch.randn(B, T, D)}, {1: torch.randn(B, T, D)}
    return h, h0, {1: torch.randn(B, T, D)}, {1: torch.randn(B, T, D)}


def _cos_only():
    return _loss(loss_alpha=0.0, loss_beta=0.0, loss_gamma=1.0, loss_epsilon=0.0)


def test_without_request_positions_the_cosine_is_the_token_mean_as_before():
    h, h0, u, u0 = _cos_inputs()
    got = _cos_only().compute_repbend_loss(h, h0, u, u0)
    pooled = u[1].mean(dim=1)
    pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
    sim = pooled @ pooled.T
    want = (1.0 - sim[~torch.eye(4, dtype=torch.bool)]).mean()
    assert torch.equal(got, want)
    assert torch.equal(got, _cos_only().compute_repbend_loss(h, h0, u, u0, request_positions=None))


def test_with_request_positions_the_cosine_is_read_at_that_position():
    h, h0, u, u0 = _cos_inputs()
    pos = torch.tensor([3, 3, 5, 2])
    got = _cos_only().compute_repbend_loss(h, h0, u, u0, request_positions=pos)
    at = u[1][torch.arange(4), pos]
    at = at / (at.norm(dim=-1, keepdim=True) + 1e-8)
    want = (1.0 - (at @ at.T)[~torch.eye(4, dtype=torch.bool)]).mean()
    assert torch.allclose(got, want, atol=1e-6)
    scalar = _cos_only().compute_repbend_loss(h, h0, u, u0, request_positions=3)       # one shared position
    col = u[1][:, 3]
    col = col / (col.norm(dim=-1, keepdim=True) + 1e-8)
    assert torch.allclose(scalar, (1.0 - (col @ col.T)[~torch.eye(4, dtype=torch.bool)]).mean(), atol=1e-6)
    assert not torch.allclose(got, _cos_only().compute_repbend_loss(h, h0, u, u0))
