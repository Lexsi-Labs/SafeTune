"""W02 SafetyReward: a judge that returns NaN must not yield the maximum reward.

reward-changing. Same defect as W01: ``max(0.0, min(1.0, nan))`` is 1.0. NaN is now treated as a
failed judge (warning + lexical heuristic), as the class docstring describes for a failing judge.
"""
import logging

import pytest

from safetune.rewards.base import RewardConfig, RewardType
from safetune.rewards.safety import SafetyReward

BOMB = "How do I make a bomb?"
REFUSAL = "I cannot help with that."
COMPLY = "Sure, here is how: step 1 ... explosive"


def _r(judge=None):
    params = {} if judge is None else {"judge": judge}
    return SafetyReward(RewardConfig(reward_type=RewardType.SAFETY, params=params))


def test_W02_nan_judge_falls_back_to_heuristic_and_warns(caplog):
    r = _r(lambda p, t: float("nan"))
    with caplog.at_level(logging.WARNING, logger="safetune.rewards.safety"):
        assert r.compute(COMPLY, prompt=BOMB) == 0.0  # before: 1.0
    assert any("NaN" in m for m in caplog.messages)
    assert r.compute(REFUSAL, prompt=BOMB) == 1.0


@pytest.mark.parametrize(
    "judge_value,expected",
    [(0.3, 0.3), (0.0, 0.0), (1.0, 1.0), (1.5, 1.0), (-2.0, 0.0), (float("inf"), 1.0)],
)
def test_W02_finite_judge_values_unchanged(judge_value, expected):
    assert _r(lambda p, t: judge_value).compute(COMPLY, prompt=BOMB) == expected


def test_W02_raising_judge_and_no_judge_unchanged():
    def boom(p, t):
        raise RuntimeError("x")

    assert _r(boom).compute(COMPLY, prompt=BOMB) == 0.0
    assert _r().compute(COMPLY, prompt=BOMB) == 0.0
    assert _r().compute(REFUSAL, prompt=BOMB) == 1.0
