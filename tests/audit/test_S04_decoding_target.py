"""Audit S04 / S06 / S05 / S16: the evaluation backend calls `processor.generate(**inputs)`, which used to run the GUIDE
(`GuidedLogitsProcessor.generate` delegated to `self.guide.generate`). It now runs the TARGET with the processor attached,
which equals the direct HF use `target.generate(logits_processor=[proc])`."""
import pytest
import torch
from transformers import LogitsProcessorList

import safetune.steer as S
from safetune.runner import steer as R

from _steer_common import HARMFUL, HARMLESS, build, prompt_ids

KW = dict(max_new_tokens=6, do_sample=False, pad_token_id=0)


def _models():
    T, tok = build(seed=0)
    E, _ = build(seed=1)
    return T, E, tok


def _direct(T, proc, ids):
    return T.generate(ids, logits_processor=LogitsProcessorList([proc]), **KW)


def test_contrastive_generate_runs_the_target_and_equals_direct_use():
    T, E, tok = _models()
    ids = prompt_ids(tok, HARMFUL[0])
    proc = R.ContrastiveDecodingTrainer(T, tok, weak_model=E, alpha=0.5).calibrate(HARMFUL, HARMLESS)[0]
    plain, weak = T.generate(ids, **KW), E.generate(ids, **KW)
    out = proc.generate(ids, **KW)
    assert not torch.equal(out, weak), "processor.generate still generates from the weak (guide) model"
    assert torch.equal(out, _direct(T, proc, ids))
    assert not torch.equal(out, plain), "the processor changed nothing"


def test_a_processor_built_without_a_target_refuses_to_generate():
    T, E, tok = _models()
    proc = S.ContrastiveDecodingProcessor(E, tok)
    with pytest.raises(ValueError, match="needs the target model"):
        proc.generate(prompt_ids(tok, HARMFUL[0]), **KW)
    # direct use needs no target and is unchanged
    ids = prompt_ids(tok, HARMFUL[0])
    assert not torch.equal(_direct(T, proc, ids), T.generate(ids, **KW))


def test_safedecoding_generate_runs_the_target():
    T, E, tok = _models()
    ids = prompt_ids(tok, HARMFUL[0])
    proc = R.SafeDecodingTrainer(T, tok, expert_model=E).calibrate(HARMFUL, HARMLESS)[0]
    out = proc.generate(ids, **KW)
    assert not torch.equal(out, E.generate(ids, **KW)), "processor.generate still generates from the expert (guide)"


# ---- S06: the window is derived per generation -----------------------------------------------------------------

def _ids_of_length(tok, n):
    ids = prompt_ids(tok, HARMFUL[0])
    return torch.cat([ids, ids, ids], dim=1)[:, :n]


def test_safedecoding_window_opens_for_each_direct_generation_whatever_the_prompt_length():
    T, E, tok = _models()
    proc = R.SafeDecodingTrainer(T, tok, expert_model=E, alpha=1.0).calibrate(HARMFUL, HARMLESS)[0]
    for n in (12, 24):                              # two generations through the same processor
        ids = _ids_of_length(tok, n)
        reference = S.SafeDecodingProcessor(E, tok, prompt_length=n, config=S.SafeDecodingConfig(alpha=1.0), target=T)
        want = _direct(T, reference, ids)
        got = _direct(T, proc, ids)
        assert not torch.equal(got, T.generate(ids, **KW)), f"window never opened for a prompt of {n} tokens"
        assert torch.equal(got, want), f"prompt of {n} tokens"


def test_safedecoding_generate_equals_direct_use_and_differs_from_the_target():
    T, E, tok = _models()
    ids = prompt_ids(tok, HARMFUL[0])
    proc = R.SafeDecodingTrainer(T, tok, expert_model=E, alpha=1.0).calibrate(HARMFUL, HARMLESS)[0]
    via_generate = proc.generate(ids, **KW)
    assert torch.equal(via_generate, _direct(T, proc, ids))
    assert not torch.equal(via_generate, T.generate(ids, **KW))


def test_an_explicit_prompt_length_is_unchanged():
    T, E, tok = _models()
    ids = prompt_ids(tok, HARMFUL[0])
    proc = S.SafeDecodingProcessor(E, tok, ids.shape[1], config=S.SafeDecodingConfig(alpha=1.0), target=T)
    assert torch.equal(proc.generate(ids, **KW), _direct(T, proc, ids))


def test_a_reused_processor_does_not_carry_the_guides_cache_into_the_next_generation():
    T, E, tok = _models()
    a, b = _ids_of_length(tok, 12), _ids_of_length(tok, 24)
    reused = S.ContrastiveDecodingProcessor(E, tok, config=S.ContrastiveDecodingConfig(alpha=0.5), target=T)
    _direct(T, reused, a)
    fresh = S.ContrastiveDecodingProcessor(E, tok, config=S.ContrastiveDecodingConfig(alpha=0.5), target=T)
    assert torch.equal(_direct(T, reused, b), _direct(T, fresh, b))
