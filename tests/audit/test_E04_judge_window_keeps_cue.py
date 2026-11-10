"""Audit E04: a judge input longer than the judge's window keeps the end of its template.

run_judge() and evaluate() cut an over-long judge input on the right. The cut removed the end of
the template (WildGuard's ``Answers: [/INST]\\n<|assistant|>``), which is what makes the classifier
answer. WildGuard then continues the response text, the parser finds no "response refusal: yes",
and the row is counted as complied. HFJudge already keeps the tail (test_E04_hfjudge_truncation);
these are the other two judge paths. Real tiny tokenizer (skipped if not cached), no judge model.
"""
import importlib

import pytest
import torch

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
CUE = "Answers: [/INST]\n<|assistant|>"
LONG = "placeholder " * 6000


@pytest.fixture
def tok():
    from transformers import AutoTokenizer
    try:
        t = AutoTokenizer.from_pretrained(TINY)
    except OSError as e:
        pytest.skip(f"{TINY} tokenizer not available offline: {type(e).__name__}")
    t.padding_side = "left"
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


def _run_judge_ids(monkeypatch, tok, prompt, response):
    """The token ids run_judge('wildguard') hands to the judge model."""
    import transformers
    import safetune
    judges = importlib.import_module("safetune.instrumentation.evaluate.judges")
    seen = {}

    def fake_hf(model_id, token_ids, max_new_tokens):
        seen["ids"] = token_ids
        return ["Harmful request: yes\nResponse refusal: yes\nHarmful response: no"] * len(token_ids)

    monkeypatch.setattr(judges, "_run_hf_judge", fake_hf)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: tok))
    before = safetune.get_config().eval_backend
    safetune.configure(eval_backend="hf")
    try:
        judges.run_judge("wildguard", [prompt], [response])
    finally:
        safetune.configure(eval_backend=before)
    return seen["ids"][0]


def test_run_judge_keeps_the_answer_cue_of_a_long_response(monkeypatch, tok):
    ids = _run_judge_ids(monkeypatch, tok, "PLACEHOLDER request", LONG)
    assert len(ids) == 4096 - 32                               # max_model_len - max_new_tokens
    assert tok.decode(ids[-30:]).rstrip().endswith(CUE), tok.decode(ids[-30:])
    text = tok.decode(ids)
    assert "You are a classifier" in text and "PLACEHOLDER request" in text    # the head survives too


def test_run_judge_leaves_a_short_input_alone(monkeypatch, tok):
    judges = importlib.import_module("safetune.instrumentation.evaluate.judges")
    ids = _run_judge_ids(monkeypatch, tok, "PLACEHOLDER request", "PLACEHOLDER response")
    want = tok.encode(judges._WILDGUARD_TEMPLATE.format(prompt="PLACEHOLDER request", response="PLACEHOLDER response"),
                      add_special_tokens=False)
    assert ids == want


class _Model:
    device = "cpu"

    def __init__(self):
        self.seen = []

    def generate(self, input_ids, attention_mask=None, **kw):
        self.seen.append((input_ids, attention_mask))
        return torch.cat([input_ids, torch.full((input_ids.shape[0], 1), 7)], dim=1)


def test_evaluate_wildguard_keeps_the_answer_cue_of_a_long_response(monkeypatch, tok):
    ev = importlib.import_module("safetune.instrumentation.evaluate.suite.evaluate")
    model = _Model()
    monkeypatch.setattr(ev, "get_wildguard", lambda: (tok, model))
    judge = ev._build_judge_scorer("wildguard")
    judge.score([ev._pack_pair("PLACEHOLDER request", LONG), ev._pack_pair("PLACEHOLDER request", "short")])

    ids, mask = model.seen[-1]
    assert ids.shape[1] == ev.get_config().max_model_len
    for row in range(2):                                        # the long row and the padded short row
        real = ids[row][mask[row].bool()]
        assert tok.decode(real[-30:]).rstrip().endswith(CUE), tok.decode(real[-30:])
    assert "You are a classifier" in tok.decode(ids[0])
