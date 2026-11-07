"""Audit S03 (SafeTune paper, arXiv 2609.22153; the paper text was not available, so this checks internal consistency).

`_collect_pooled` used the tokenizer's own padding side; with right padding "last_token" was a pad position, so the
score of a prompt depended on its batch. (`generate` / `__call__` are reported separately, not changed.)
"""
import torch

import safetune.steer as S

from _steer_common import HARMFUL, HARMLESS, build


def _guard():
    m, tok = build()
    probe = S.fit_linear_probe(m, tok, HARMFUL * 5, HARMLESS * 5, S.LinearProbeConfig(pick_layer=3))
    return m, tok, S.LinearProbeGuardModel(m, tok, probe)


def test_score_does_not_depend_on_the_batch_with_a_right_padding_tokenizer():
    m, tok, g = _guard()
    tok.padding_side = "right"
    prompts = HARMFUL + HARMLESS
    s1 = torch.cat([g.score([p]) for p in prompts])
    s8 = g.score(prompts)
    assert (s1 - s8).abs().max() < 1e-4, f"batch 1 vs 16: max score diff {(s1 - s8).abs().max():.3f}"


def test_right_padded_scores_equal_left_padded_scores():
    m, tok, g = _guard()
    prompts = HARMFUL + HARMLESS
    tok.padding_side = "left"
    left = g.score(prompts)
    tok.padding_side = "right"
    right = g.score(prompts)
    assert torch.allclose(left, right, atol=1e-4), f"max diff {(left - right).abs().max():.4f}"


def test_left_padded_last_token_pool_is_unchanged():
    from safetune.interventions.steer.probe_guard import _pool_hidden
    h = torch.randn(3, 5, 8)
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert torch.equal(_pool_hidden(h, "last_token", mask), h[:, -1, :])
    assert torch.equal(_pool_hidden(h, "mean", torch.ones(3, 5, dtype=torch.long)), h.mean(dim=1))
