"""Audit E06: ScorerJudgeAdapter. Mock judges only."""
import pytest

from safetune.core.eval.judge_adapters import ScorerJudgeAdapter


class _J:
    def __init__(self, judgement=None):
        self.rows, self.judgement = [], judgement or {"asr": 1.0}

    def score(self, rows):
        self.rows += rows
        return [{**r, "judgement": dict(self.judgement)} for r in rows]


def test_a_judge_that_returns_no_asr_key_raises():
    with pytest.raises(ValueError, match="asr"):
        ScorerJudgeAdapter(_J({"verdict": "unsafe"})).score(["r1", "r2"])


def test_fewer_behaviors_than_texts_raises():
    j = _J()
    with pytest.raises(ValueError, match="1 behaviors for 3 texts"):
        ScorerJudgeAdapter(j, behaviors=["only one"]).score(["r1", "r2", "r3"])
    assert j.rows == [], "the judge was called before the length check"


def test_behaviors_none_stays_allowed_and_scores_are_unchanged():
    """Already-correct paths: documented behaviors=None, and equal-length behaviors."""
    j = _J({"asr": 0.25})
    assert ScorerJudgeAdapter(j).score(["a", "b"]) == [0.25, 0.25]
    assert [r["prompt"] for r in j.rows] == ["", ""]
    j = _J({"asr": 1.0})
    assert ScorerJudgeAdapter(j, behaviors=["p1", "p2"]).score(["a", "b"]) == [1.0, 1.0]
    assert [r["prompt"] for r in j.rows] == ["p1", "p2"]
    assert ScorerJudgeAdapter(_J({"x": 3.0}), score_key="x").score(["a"]) == [3.0]
