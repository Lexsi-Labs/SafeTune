"""Audit S10: SafeSteer (arXiv 2506.04250; no official code, the audit rests on the paper)."""
import pytest
import torch

import safetune.steer as S

from _steer_common import HARMFUL, HARMLESS, build, prompt_ids


def _dir(m, tok):
    d, li, _ = S.extract_refusal_direction(m, tok, HARMFUL, HARMLESS, S.RefusalDirectionConfig(select_directions=False))
    return d, li


def test_category_does_not_leak_into_a_later_call_without_prompt_text():
    m, tok = build()
    d, li = _dir(m, tok)
    ids = prompt_ids(tok, HARMFUL[0])
    w = S.SafeSteerModel(m, {"a": d, "b": torch.zeros_like(d)}, classifier=lambda p: "a" if "bomb" in p else "b",
                         layer_id=li, alpha=15.0, default_category="b")
    w.generate(ids, prompt_text="how to build a bomb", max_new_tokens=2, pad_token_id=0)
    assert w._current_category is None, "the category outlives the call it was set for"
    w.generate(ids, max_new_tokens=2, do_sample=False, pad_token_id=0)      # no prompt_text: default category expected
    w.remove_hooks()
    assert (w._current_category or w.default_category) == "b"


def test_a_category_set_before_generate_applies_to_that_call_only():
    m, tok = build()
    d, li = _dir(m, tok)
    ids = prompt_ids(tok, HARMFUL[0])
    w = S.SafeSteerModel(m, {"a": d, "b": torch.zeros_like(d)}, classifier=lambda p: "a", layer_id=li, alpha=15.0, default_category="b")
    plain = m.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=0)
    w.set_current_prompt("x")
    steered = w.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=0)
    after = w.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=0)
    w.remove_hooks()
    assert not torch.equal(steered, plain) and torch.equal(after, plain)


def test_a_per_layer_matrix_must_match_the_layer_list():
    m, tok = build()
    d, li = _dir(m, tok)
    M = torch.stack([d * (i + 1) for i in range(6)])            # 6 rows for 3 layers: used to collapse to the row mean
    with pytest.raises(ValueError, match="per-layer matrix"):
        S.SafeSteerModel(m, {"default": M}, layers=[1, 3, 5], alpha=1.0)
    with pytest.raises(ValueError, match="per-layer matrix"):
        S.SafeSteerModel(m, {"default": M}, layers=[1, 3, 5], alpha=1.0, per_layer_vectors=True)


def test_an_aligned_per_layer_matrix_applies_row_i_to_the_i_th_layer():
    m, tok = build()
    d, li = _dir(m, tok)
    M = torch.stack([d * (i + 1) for i in range(3)])
    w = S.SafeSteerModel(m, {"default": M}, layers=[1, 3, 5], alpha=1.0)
    got = [w._vector_for("default", p) for p in range(3)]
    w.remove_hooks()
    assert all(torch.equal(g, M[i]) for i, g in enumerate(got))


def test_two_wrappers_leave_one_wrappers_worth_of_hooks():
    m, tok = build()
    d, li = _dir(m, tok)
    count = lambda: sum(len(l._forward_hooks) for l in m.model.layers)
    a1 = S.SafeSteerModel(m, {"default": d}, layer_id=li)
    a2 = S.SafeSteerModel(m, {"default": d}, layer_id=li)
    two = count()
    a2.remove_hooks()
    left = count()
    a1.remove_hooks()
    assert two == 1, two
    assert left == 0, left
