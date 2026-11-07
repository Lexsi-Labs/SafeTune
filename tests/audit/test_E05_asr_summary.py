"""Audit E05: asr_summary vs strongreject@f7cad6c strongreject_evaluator.py (an unscored row is NaN, not 0)."""
import math

from safetune.core.eval.pipeline.scorer import asr_summary


def test_unscored_rows_are_left_out_and_counted():
    s = asr_summary([{"judgement": {"asr": 1.0}}, {"judgement": {"asr": float("nan")}}, {"response": "r"},
                     {"judgement": {"verdict": "x"}}, {"judgement": None}])
    assert s == {"n": 1, "asr": 1.0, "refusal_rate": 0.0, "n_unscored": 4, "n_unparsed": 0}


def test_all_unscored_or_empty_reports_zero_rows_not_a_safe_model():
    assert asr_summary([{"response": "r"}]) == {"n": 0, "asr": 0.0, "refusal_rate": 0.0, "n_unscored": 1, "n_unparsed": 0}
    assert asr_summary([]) == {"n": 0, "asr": 0.0, "refusal_rate": 0.0, "n_unscored": 0, "n_unparsed": 0}


def test_fully_scored_rows_are_bit_identical_to_the_old_summary():
    rows = [{"judgement": {"asr": a}} for a in (0.0, 1.0, 0.25, 0.75, 0.0)]
    old = sum(r["judgement"]["asr"] for r in rows) / 5
    s = asr_summary(rows)
    assert s["n"] == 5 and s["asr"] == old and s["refusal_rate"] == 1.0 - old and s["n_unscored"] == 0
    assert not any(math.isnan(v) for v in s.values())
