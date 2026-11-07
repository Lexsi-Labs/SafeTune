"""Audit E13: XSTest over-refusal rate. XSTest = 250 safe prompts (answering is right) + 200 unsafe contrast prompts (refusing is right).
Fake benchmark loader and judge; the label values are checked against code only (walledai/XSTest is gated, its card is not readable)."""
import importlib

import pytest
from datasets import Dataset

SB = importlib.import_module("safetune.instrumentation.evaluate.suite.benchmarks")
E = importlib.import_module("safetune.instrumentation.evaluate.suite.evaluate")


def _xstest_run(monkeypatch, rows, refuse_unsafe=True):
    monkeypatch.setattr(SB, "load_benchmark", lambda name: Dataset.from_list(rows))
    monkeypatch.setattr(SB, "check_benchmarks", lambda names: None)
    seen = []

    def build(name):
        class J:
            def score(self, texts):
                prompts = [t.split(E._PAIR_SEP)[0] for t in texts]
                seen.extend(prompts)
                # 1.0 = complied (asr). The mock model complies with every safe prompt and refuses every unsafe one.
                return [0.0 if "unsafe" in p else 1.0 for p in prompts]
        return J()
    monkeypatch.setattr(E, "_build_judge_scorer", build)

    class M:                         # no tokenizer: evaluate() calls model(p)
        def __call__(self, p):
            return "PLACEHOLDER answer"
    return E.evaluate(M(), benchmarks=["xstest"], judge="wildguard", batch_size=4)["xstest"], seen


XROWS = ([{"prompt": "PLACEHOLDER safe %d" % i, "label": "safe"} for i in range(3)]
         + [{"prompt": "PLACEHOLDER unsafe %d" % i, "label": "unsafe"} for i in range(2)])


def test_xstest_headline_is_the_safe_subset_and_both_subsets_are_reported(monkeypatch):
    out, seen = _xstest_run(monkeypatch, XROWS)
    assert out["headline_metric"] == "refusal_rate"
    assert out["refusal_rate"] == 0.0 and out["asr"] == 1.0 and out["n_evaluated"] == 3      # safe only: nothing over-refused
    assert out["safe"]["n_evaluated"] == 3 and out["unsafe"]["refusal_rate"] == 1.0
    assert (out["n_safe"], out["n_unsafe"], out["n_total"]) == (3, 2, 5)


def test_xstest_without_a_usable_label_raises_instead_of_pooling(monkeypatch):
    with pytest.raises(ValueError, match="label"):
        _xstest_run(monkeypatch, [{"prompt": "PLACEHOLDER a"}, {"prompt": "PLACEHOLDER b"}])
    with pytest.raises(ValueError, match="label"):
        _xstest_run(monkeypatch, [{"prompt": "PLACEHOLDER a", "label": "maybe"}])


def test_other_benchmarks_are_scored_exactly_as_before(monkeypatch):
    rows = [{"prompt": "PLACEHOLDER p%d" % i} for i in range(3)]
    monkeypatch.setattr(SB, "load_benchmark", lambda name: Dataset.from_list(rows))
    monkeypatch.setattr(SB, "check_benchmarks", lambda names: None)
    monkeypatch.setattr(E, "_build_judge_scorer", lambda name: type("J", (), {"score": lambda self, t: [1.0, 0.0, 1.0]})())

    class M:
        def __call__(self, p):
            return "PLACEHOLDER answer"
    out = E.evaluate(M(), benchmarks=["advbench"], judge="wildguard", batch_size=4)["advbench"]
    assert out["asr"] == 2 / 3 and "safe" not in out and out["n_evaluated"] == 3
