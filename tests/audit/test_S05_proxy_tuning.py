"""Audit S05: Proxy Tuning (alisawuffles/proxy-tuning@5f2da2c, modeling/dexperts.py:155-170).

`proxy_base` was called without an attention mask, so on a left-padded batch it attended the pad tokens while the tuned proxy
did not, and the delta `tuned - base` was wrong for the padded rows.
"""
import pytest
import torch
from transformers import LogitsProcessorList

import safetune.steer as S
from safetune.runner import steer as R

from _steer_common import HARMFUL, build, prompt_ids


def _setup():
    T, tok = build(seed=0)
    tuned, _ = build(seed=1)
    base, _ = build(seed=2)
    tok.padding_side = "left"
    return T, tuned, base, tok


def _padded_batch(tok):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for p in ("Hi", HARMFUL[0])]
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)


def test_a_left_padded_row_gets_the_same_delta_as_the_unpadded_row():
    T, tuned, base, tok = _setup()
    e = _padded_batch(tok)
    row0 = e["input_ids"][0][e["attention_mask"][0].bool()].unsqueeze(0)
    zeros = lambda n: torch.zeros(n, 32000)
    batch = S.ProxyTuningProcessor(tuned, base, tok)(e["input_ids"], zeros(2))
    single = S.ProxyTuningProcessor(tuned, base, tok)(row0, zeros(1))
    assert (batch[0] - single[0]).abs().max() < 1e-4, f"max diff {(batch[0] - single[0]).abs().max():.4f}"


def test_an_unpadded_batch_is_bit_identical_to_the_call_without_a_mask():
    T, tuned, base, tok = _setup()
    ids = prompt_ids(tok, HARMFUL[0])
    proc = S.ProxyTuningProcessor(tuned, base, tok)
    got = proc._proxy_base_logits(ids)
    with torch.no_grad():
        want = base(input_ids=ids, use_cache=False).logits[:, -1, :].float()
    assert torch.equal(got, want)


def test_generate_runs_the_target_and_equals_direct_use():
    T, tuned, base, tok = _setup()
    ids = prompt_ids(tok, HARMFUL[0])
    kw = dict(max_new_tokens=6, do_sample=False, pad_token_id=0)
    proc = S.ProxyTuningProcessor(tuned, base, tok, config=S.ProxyTuningConfig(scale=1.0), target=T)
    via_generate = proc.generate(ids, **kw)
    assert torch.equal(via_generate, T.generate(ids, logits_processor=LogitsProcessorList([proc]), **kw))
    assert not torch.equal(via_generate, T.generate(ids, **kw))


# ---- the runner trainer ------------------------------------------------------------------------------------------

def test_trainer_raises_without_both_proxies():
    T, tuned, base, tok = _setup()
    with pytest.raises(ValueError, match="proxy_tuned_model"):
        R.ProxyTuningTrainer(T, tok).calibrate()
    with pytest.raises(ValueError, match="proxy_base_model"):
        R.ProxyTuningTrainer(T, tok, proxy_tuned_model=tuned).calibrate()
    with pytest.raises(ValueError, match="proxy_tuned_model"):
        R.ProxyTuningTrainer(T, tok, proxy_base_model=base).calibrate()


def test_trainer_rejects_the_old_proxy_model_argument_and_names_the_new_ones():
    T, tuned, base, tok = _setup()
    with pytest.raises(ValueError, match=r"proxy_model is gone.*proxy_tuned_model.*proxy_base_model"):
        R.ProxyTuningTrainer(T, tok, proxy_model=base)


def test_trainer_generate_differs_from_the_target_and_equals_direct_use():
    T, tuned, base, tok = _setup()
    kw = dict(max_new_tokens=6, do_sample=False, pad_token_id=0)
    e = _padded_batch(tok)
    proc = R.ProxyTuningTrainer(T, tok, proxy_tuned_model=tuned, proxy_base_model=base).calibrate()[0]
    via_generate = proc.generate(**e, **kw)
    assert torch.equal(via_generate, T.generate(**e, logits_processor=LogitsProcessorList([proc]), **kw))
    assert not torch.equal(via_generate, T.generate(**e, **kw))
    # the delta is tuned - base added to the target, not the other way round
    ref = S.ProxyTuningProcessor(tuned, base, tok, target=T)
    assert torch.equal(via_generate, T.generate(**e, logits_processor=LogitsProcessorList([ref]), **kw))


def test_trainer_rejects_one_model_given_as_both_proxies():
    T, tuned, base, tok = _setup()
    with pytest.raises(ValueError, match="same object"):
        R.ProxyTuningTrainer(T, tok, proxy_tuned_model=tuned, proxy_base_model=tuned).calibrate()
