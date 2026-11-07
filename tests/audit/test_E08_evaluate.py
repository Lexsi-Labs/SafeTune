"""Audit E08: evaluate() (suite/evaluate.py). Tiny random Llama, mocked benchmark loader and judge, placeholder strings."""
import importlib

import pytest
import torch

import safetune.instrumentation.evaluate.suite.benchmarks as B
from safetune.evaluate import evaluate

from _eval_common import build

E = importlib.import_module("safetune.instrumentation.evaluate.suite.evaluate")   # the package re-exports the function under the same name

ROWS = [{"prompt": "PLACEHOLDER prompt %d" % i} for i in range(6)]


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(B, "load_benchmark", lambda name: ROWS)
    monkeypatch.setattr(B, "check_benchmarks", lambda names: None)
    state = {}

    def build_scorer(name):
        class J:
            def score(self, texts):
                state["texts"] = texts
                return [0.0 for _ in texts]          # nothing is unsafe / harmful
        return J()
    monkeypatch.setattr(E, "_build_judge_scorer", build_scorer)
    return state


def _spy(model):
    seen = []
    orig = model.generate

    def gen(*a, **kw):
        seen.append(kw)
        return orig(*a, **kw)
    model.generate = gen
    return seen


def test_hf_generation_is_greedy_unless_the_caller_says_otherwise(patched):
    """The vLLM path and run_safety_eval pass temperature=0.0; the HF path leaves do_sample to the model's generation_config (sampling for most instruct models)."""
    m, tok = build()
    m.generation_config.do_sample = True
    seen = _spy(m)
    evaluate(m, benchmarks=["advbench"], judge="harmbench", tokenizer=tok, batch_size=3, max_new_tokens=4)
    assert seen and all(kw.get("do_sample") is False for kw in seen), f"generate kwargs: {seen[0].keys()}"


def test_a_caller_supplied_do_sample_still_wins(patched):
    m, tok = build()
    seen = _spy(m)
    evaluate(m, benchmarks=["advbench"], judge="harmbench", tokenizer=tok, batch_size=3, max_new_tokens=4,
             generation_kwargs={"do_sample": True, "temperature": 0.7})
    assert all(kw["do_sample"] is True and kw["temperature"] == 0.7 for kw in seen)


def test_results_line_up_with_the_input_rows_across_batches(patched):
    m, tok = build()
    evaluate(m, benchmarks=["advbench"], judge="harmbench", tokenizer=tok, batch_size=4, max_new_tokens=2)
    last = patched["texts"]                       # the last batch: rows 4 and 5, in order
    assert [t.split(E._PAIR_SEP)[0] for t in last] == [r["prompt"] for r in ROWS[4:]]


@pytest.mark.parametrize("judge", ["harmbench", "llama_guard_3"])
@pytest.mark.parametrize("bench", ["xstest", "orbench", "orbench_hard"])
def test_over_refusal_benchmark_with_a_harm_judge_is_rejected(patched, judge, bench):
    """xstest/orbench headline_metric is refusal_rate = 1 - asr, but a harm classifier says 'not harmful' for every benign prompt: a model that answers everything gets refusal_rate 1.0."""
    m, tok = build()
    with pytest.raises(ValueError, match="wildguard"):
        evaluate(m, benchmarks=[bench], judge=judge, tokenizer=tok, batch_size=3, max_new_tokens=2)


def test_over_refusal_with_wildguard_and_harm_benchmarks_with_any_judge_still_run(patched):
    """Already-correct paths: wildguard on orbench_hard; harmbench/llama_guard_3 on harm benchmarks and orbench_toxic."""
    m, tok = build()
    assert evaluate(m, benchmarks=["orbench_hard"], judge="wildguard", tokenizer=tok, batch_size=3, max_new_tokens=2)["orbench_hard"]["headline_metric"] == "refusal_rate"
    for judge in ("harmbench", "llama_guard_3"):
        out = evaluate(m, benchmarks=["advbench", "orbench_toxic"], judge=judge, tokenizer=tok, batch_size=3, max_new_tokens=2)
        assert out["advbench"]["headline_metric"] == out["orbench_toxic"]["headline_metric"] == "asr"


def test_an_empty_benchmark_has_no_asr_and_no_refusal_rate():
    """run_safety_eval path: zero scored prompts used to read as asr 0.0 / refusal_rate 1.0."""
    from safetune.instrumentation.evaluate.pipeline import _compute_bench_metrics
    m = _compute_bench_metrics([])
    assert m["n"] == 0 and m["asr"] != m["asr"] and m["refusal_rate"] != m["refusal_rate"]


def test_scored_benchmark_metrics_are_unchanged():
    from safetune.instrumentation.evaluate.pipeline import _compute_bench_metrics
    assert _compute_bench_metrics([0.0, 1.0, 1.0, 0.0]) == {"n": 4, "asr": 0.5, "refusal_rate": 0.5, "scores": [0.0, 1.0, 1.0, 0.0]}
