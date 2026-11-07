"""Audit S12: SCANS (zouyingcao/SCANS@9dc20d2).

Official: the anchor vectors, the reference transition and every classification use the chat-formatted text
(`[INST] q [/INST]`, then ` Sure`; SCANS_llama.py:47-89), and the sign is set per prompt before every generation
(`:137-145`).
"""
import pytest
import torch

import safetune.steer as S

from _steer_common import HARMFUL, HARMLESS, build


def _templated(tok, q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)


def test_default_fit_is_the_plain_text_computation_unchanged():
    m, tok = build()
    w = S.SCANSModel(m, tokenizer=tok)
    w.fit(HARMFUL, HARMLESS)
    assert w._chat_template is False
    w.remove_hooks()                       # fit() arms the steering hooks; recompute on the unsteered model
    hs = lambda t: w._hidden_states(t)
    for layer in (1, 3, 5):
        neg = torch.stack([hs(q)[layer + 1, -1] for q in HARMFUL]).mean(0)
        pos = torch.stack([hs(q)[layer + 1, -1] for q in HARMLESS]).mean(0)
        v = neg - pos
        assert torch.equal(w._steering_vectors[layer], v / torch.norm(v))
        trans = torch.stack([hs(f"{q} Sure")[layer + 1, -2] - hs(f"{q} Sure")[layer + 1, -1] for q in HARMFUL]).mean(0)
        assert torch.equal(w._reference_harm[layer], trans)
    # classification of a prompt through the default path
    ref = S.SCANSModel(m, tokenizer=tok)
    ref.fit(HARMFUL, HARMLESS, chat_template=False)
    assert w.predict_safety(HARMLESS[0]) == ref.predict_safety(HARMLESS[0])


def test_chat_template_fit_builds_everything_from_the_templated_text():
    m, tok = build()
    plain = S.SCANSModel(m, tokenizer=tok)
    plain.fit(HARMFUL, HARMLESS)
    w = S.SCANSModel(m, tokenizer=tok)
    w.fit(HARMFUL, HARMLESS, chat_template=True)
    assert w._chat_template is True
    w.remove_hooks()
    hs = lambda t: w._hidden_states(t)
    layer = 3
    neg = torch.stack([hs(_templated(tok, q))[layer + 1, -1] for q in HARMFUL]).mean(0)
    pos = torch.stack([hs(_templated(tok, q))[layer + 1, -1] for q in HARMLESS]).mean(0)
    v = neg - pos
    assert torch.allclose(w._steering_vectors[layer], v / torch.norm(v), atol=1e-6)
    t = lambda q: hs(f"{_templated(tok, q)} Sure")[layer + 1]
    ref = torch.stack([t(q)[-2] - t(q)[-1] for q in HARMFUL]).mean(0)
    assert torch.allclose(w._reference_harm[layer], ref, atol=1e-6)
    assert not torch.allclose(w._reference_harm[layer], plain._reference_harm[layer])


# ---- the per-row sign inside generate ----------------------------------------------------------------------------

PROMPTS = HARMFUL + HARMLESS


def _fitted(chat_template=True, side="left"):
    m, tok = build()
    w = S.SCANSModel(m, tokenizer=tok, multiplier=300.0)
    w.fit(HARMFUL, HARMLESS, chat_template=chat_template)
    # classification forwards run with the steering hooks armed (as in the prompt= path), so the
    # threshold is taken in that regime too
    raw = sorted(w.predict_safety(q)[0] for q in PROMPTS)
    w.threshold = float((raw[7] + raw[8]) / 2)               # splits the prompts into two groups
    tok.padding_side = side
    return m, tok, w


def _inputs(tok, prompts, chat_template=True):
    texts = [_templated(tok, q) if chat_template else q for q in prompts]
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=not chat_template)


def _signs_seen_by_generate(m, w, **inputs):
    seen = {}
    def grab(mod, a, kw):              # the classification forwards run before the signs are set
        if w._row_sign is not None:
            seen.setdefault("s", w._row_sign.clone())
    h = m.register_forward_pre_hook(grab, with_kwargs=True)
    w.generate(**inputs, max_new_tokens=2, do_sample=False, pad_token_id=0)
    h.remove()
    return seen["s"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_generate_on_templated_inputs_agrees_with_the_prompt_path_on_all_16_prompts(side):
    m, tok, w = _fitted(True, side)
    expected = torch.tensor([w.predict_safety(q)[1] for q in PROMPTS])
    assert (expected > 0).any() and (expected < 0).any(), "premise: both signs occur"
    got = _signs_seen_by_generate(m, w, **_inputs(tok, PROMPTS))
    rate = float((got == expected).float().mean())
    print(f"agreement (padding={side}): {int((got == expected).sum())}/16")
    assert rate == 1.0, f"agreement {rate:.3f}"


def test_a_mixed_batch_gets_one_sign_per_row_and_keeps_no_state():
    m, tok, w = _fitted(True)
    prompts = [HARMFUL[0], HARMLESS[0], HARMFUL[1], HARMLESS[1]]
    expected = torch.tensor([w.predict_safety(q)[1] for q in prompts])
    got = _signs_seen_by_generate(m, w, **_inputs(tok, prompts))
    assert torch.equal(got, expected)
    assert w._row_sign is None and w._current_sign == 1.0
    w.remove_hooks()


def test_batch_of_one_matches_the_prompt_path_bit_for_bit():
    m, tok, w = _fitted(True)
    for q in (HARMFUL[0], HARMLESS[0]):
        e = _inputs(tok, [q])
        via_inputs = w.generate(**e, max_new_tokens=4, do_sample=False, pad_token_id=0)
        via_prompt = w.generate(**e, prompt=q, max_new_tokens=4, do_sample=False, pad_token_id=0)
        assert torch.equal(via_inputs, via_prompt)
    w.remove_hooks()


def test_the_prompt_path_is_unchanged():
    m, tok, w = _fitted(True)
    e = _inputs(tok, [HARMLESS[0]])
    out = w.generate(**e, prompt=HARMLESS[0], max_new_tokens=4, do_sample=False, pad_token_id=0)
    w.set_prompt(HARMLESS[0])                                   # the old flow: arm the sign, call the model directly
    old = m.generate(**e, max_new_tokens=4, do_sample=False, pad_token_id=0)
    w.remove_hooks()
    assert torch.equal(out, old)


def test_plain_text_fit_still_classifies_plain_tokenized_rows():
    m, tok, w = _fitted(False)
    expected = torch.tensor([w.predict_safety(q)[1] for q in PROMPTS])
    got = _signs_seen_by_generate(m, w, **_inputs(tok, PROMPTS, chat_template=False))
    assert torch.equal(got, expected)
    w.remove_hooks()


def test_scans_trainer_uses_the_chat_template_when_the_tokenizer_has_one():
    from safetune.runner.steer import SCANSTrainer
    m, tok = build()
    w, _ = SCANSTrainer(m, tok).calibrate(HARMFUL, HARMLESS)
    assert w._chat_template is True
    w.remove_hooks()
    w2, _ = SCANSTrainer(m, tok, chat_template=False).calibrate(HARMFUL, HARMLESS)
    assert w2._chat_template is False
    w2.remove_hooks()


def test_per_row_signs_change_the_output_against_the_unsteered_model():
    m, tok, w = _fitted(True)
    e = _inputs(tok, PROMPTS)
    kw = dict(max_new_tokens=5, do_sample=False, pad_token_id=0)
    steered = w.generate(**e, **kw)
    w.remove_hooks()
    plain = m.generate(**e, **kw)
    assert not torch.equal(steered, plain), "SCANS steering changed no row"
