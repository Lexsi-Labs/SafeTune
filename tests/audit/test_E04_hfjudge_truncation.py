"""Audit E04: HFJudge clips the GENERATION before templating, as HarmBench@8e1604d evaluate_completions.py:54-62 does (right side, 512 tokens),
and truncates an over-long prompt from the LEFT (evaluate_completions.py:66), so `Answer: [/INST]` is always the end of the classifier input.
Real tiny tokenizer (skipped if not cached), fake classifier model; no network."""
import pytest
import torch

from safetune.core.eval.pipeline.scorer import HFJudge, HFJudgeConfig

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
LLAMA = "cais/HarmBench-Llama-2-13b-cls"


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.seen = []

    def generate(self, input_ids, attention_mask=None, **kw):
        self.seen.append(input_ids[0].tolist())
        return torch.cat([input_ids, torch.tensor([[7]])], dim=1)


@pytest.fixture
def judge():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(TINY)
    except OSError as e:
        pytest.skip(f"{TINY} tokenizer not available offline: {type(e).__name__}")
    j = HFJudge(HFJudgeConfig(model=LLAMA))
    j._tokenizer, j._model = tok, _Model()
    return j


def _tail_and_text(j):
    ids = j._model.seen[-1]
    return j._tokenizer.decode(ids[-14:]), j._tokenizer.decode(ids)


def test_a_long_generation_is_clipped_so_the_instruction_tail_is_present(judge):
    judge._judge_one("PLACEHOLDER behavior", "placeholder " * 6000)
    tail, text = _tail_and_text(judge)
    assert "Answer: [/INST]" in tail, tail
    assert text.count("placeholder") < 700            # about 512 tokens, not 6000
    assert "[INST]" in text and "Rules:" in text       # the head of the template survives too


def test_a_huge_behavior_drops_the_left_and_keeps_the_tail(judge):
    judge._judge_one("placeholder " * 6000, "PLACEHOLDER generation")
    tail, _ = _tail_and_text(judge)
    assert "Answer: [/INST]" in tail, tail
    assert len(judge._model.seen[-1]) <= 3000


def test_a_short_generation_reaches_the_classifier_unchanged(judge):
    from safetune.instrumentation.evaluate.judges._templates import harmbench_template
    judge._judge_one("PLACEHOLDER behavior", "PLACEHOLDER generation")
    expect = harmbench_template(LLAMA).format(behavior="PLACEHOLDER behavior", generation="PLACEHOLDER generation")
    got = judge._tokenizer.decode(judge._model.seen[-1], skip_special_tokens=True)
    want = judge._tokenizer.decode(judge._tokenizer(expect, return_tensors="pt")["input_ids"][0], skip_special_tokens=True)
    assert got == want


def test_the_clip_length_is_configurable_and_can_be_switched_off(judge):
    judge.config.generation_clip_tokens = 20
    judge._judge_one("PLACEHOLDER behavior", "placeholder " * 500)
    assert _tail_and_text(judge)[1].count("placeholder") < 30
    judge.config.generation_clip_tokens = 0
    judge._judge_one("PLACEHOLDER behavior", "placeholder " * 500)
    assert _tail_and_text(judge)[1].count("placeholder") >= 400
