"""Audit E09: TamperBenchEvaluator vs criticalml-uw/TamperBench@ca4fade. A fake `strong_reject` module; no model, no network."""
import sys
import types

import pytest

from safetune.evaluate import TamperBenchEvaluator


@pytest.fixture
def fake_strong_reject(monkeypatch):
    calls = []

    def strongreject_finetuned(batch, max_response_length=512, **kw):
        calls.append((dict(batch), max_response_length))
        return {"score": [0.25 * (i % 5) for i, _ in enumerate(batch["response"])]}

    pkg = types.ModuleType("strong_reject")
    ev = types.ModuleType("strong_reject.evaluate")
    ev.strongreject_finetuned = strongreject_finetuned
    pkg.evaluate = ev
    monkeypatch.setitem(sys.modules, "strong_reject", pkg)
    monkeypatch.setitem(sys.modules, "strong_reject.evaluate", ev)
    return calls


def test_score_is_the_mean_strongreject_score_batched_like_the_harness(fake_strong_reject):
    pairs = [("PLACEHOLDER p%d" % i, "PLACEHOLDER r%d" % i) for i in range(5)]
    out = TamperBenchEvaluator().score_responses(pairs, batch_size=2)
    assert out["num_evaluated"] == 5 and abs(out["strong_reject_score"] - 0.1) < 1e-12
    assert [c[1] for c in fake_strong_reject] == [512, 512, 512]          # harness default max_response_length
    assert fake_strong_reject[0][0]["forbidden_prompt"] == ["PLACEHOLDER p0", "PLACEHOLDER p1"]


def test_attack_names_are_harness_names_and_the_unexposed_evals_are_documented():
    import re
    try:
        src = open("/home/deeraj/lexsi/safetune/refs/TamperBench/src/tamperbench/whitebox/utils/names.py").read()
    except OSError:
        pytest.skip("TamperBench clone not present")
    attacks = set(re.findall(r'^\s+[A-Z_]+ = "([a-z_]+)"', src.split("class DefenseName")[0], re.M))
    evals = set(re.findall(r'^\s+[A-Z_0-9]+ = "([a-z_0-9]+)"', src.split("class EvalName")[1].split("class MetricName")[0], re.M))
    assert set(TamperBenchEvaluator.SUPPORTED_ATTACKS) <= attacks
    doc = sys.modules[TamperBenchEvaluator.__module__].__doc__
    for name in sorted(evals - set(TamperBenchEvaluator.SUPPORTED_EVALS)):
        assert name in doc, f"{name} is rejected by the wrapper and not documented"
    assert "core.eval.tamperbench" in doc


def test_scoring_nothing_is_not_a_perfect_score(fake_strong_reject):
    out = TamperBenchEvaluator().score_responses([])
    s = out["strong_reject_score"]
    assert s != s and out["num_evaluated"] == 0, f"no pairs scored, strong_reject_score={s}"
