"""Audit S01: `RefusalDirectionTrainer` returns a `RefusalDirectionModel` that had no `generate`, was not callable and installed no
hooks, so the evaluation backend (`hasattr(model, "generate")`, else `model(p)`) failed every batch ("every batch failed to score").
It now steers for the duration of a `generate` / call, like `CAAModel`."""
import torch

import safetune.steer as S
from safetune.runner import steer as R

from _steer_common import HARMFUL, HARMLESS, build, prompt_ids

KW = dict(max_new_tokens=6, do_sample=False, pad_token_id=0)


def _count(m):
    return sum(len(l._forward_hooks) + len(l._forward_pre_hooks) + len(l.self_attn._forward_hooks) + len(l.mlp._forward_hooks)
               for l in m.model.layers)


def test_runner_wrapper_generate_is_steered_and_leaves_no_hooks():
    m, tok = build()
    ids = prompt_ids(tok, HARMFUL[0])
    plain = m.generate(ids, **KW)
    w = R.RefusalDirectionTrainer(m, tok, alpha=20.0).calibrate(HARMFUL, HARMLESS, calib_n=8)[0]
    assert hasattr(w, "generate") and callable(w) and hasattr(w, "device")
    out = w.generate(ids, **KW)
    assert not torch.equal(out, plain), "the wrapper's generate is unsteered"
    assert _count(m) == 0, "hooks left installed after generate"
    with w:                                              # the README flow is unchanged
        assert torch.equal(m.generate(ids, **KW), out)


def test_call_is_steered_too_and_install_semantics_are_unchanged():
    m, tok = build()
    ids = prompt_ids(tok, HARMFUL[0])
    d, _, _ = S.extract_refusal_direction(m, tok, HARMFUL, HARMLESS, S.RefusalDirectionConfig(select_directions=False))
    w = S.RefusalDirectionModel(m, d, mode="steer", strength=5.0)
    with torch.no_grad():
        base = m(input_ids=ids).logits
        steered = w(input_ids=ids).logits
    assert (steered - base).abs().max() > 1e-3
    assert _count(m) == 0
    w.install()
    with torch.no_grad():
        assert torch.equal(w(input_ids=ids).logits, steered)       # already installed: called as is
    assert _count(m) > 0
    w.remove()
