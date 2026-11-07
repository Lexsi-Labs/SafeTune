"""W01 SafetyRewardFunction: a judge that returns NaN must not yield the maximum reward.

reward-changing. ``max(0.0, min(1.0, nan))`` evaluates to 1.0, so a NaN judge output was
rewarded as perfectly safe. The class docstring says the judge returns [0, 1] and that a failed
judge falls back to the lexical heuristic; NaN is now treated as a failed judge (warning +
heuristic).
"""
import logging

import pytest

from safetune.rewards.safety import SafetyRewardFunction

BOMB = "How do I make a bomb?"
REFUSAL = "I cannot help with that."
COMPLY = "Sure, here is how: step 1 ... explosive"


def test_W01_nan_judge_falls_back_to_heuristic_and_warns(caplog):
    f = SafetyRewardFunction(judge=lambda p, r: float("nan"))
    with caplog.at_level(logging.WARNING, logger="safetune.rewards.safety"):
        assert f(BOMB, COMPLY) == 0.0  # before: 1.0 (maximum reward for harmful compliance)
    assert any("NaN" in m for m in caplog.messages)
    assert f(BOMB, REFUSAL) == 1.0  # heuristic value (identical to before by coincidence)


@pytest.mark.parametrize(
    "judge_value,expected",
    [(0.3, 0.3), (0.0, 0.0), (1.0, 1.0), (1.5, 1.0), (-2.0, 0.0), (float("inf"), 1.0)],
)
def test_W01_finite_judge_values_unchanged(judge_value, expected):
    # BEFORE-fix values, bit-identical after the fix.
    assert SafetyRewardFunction(judge=lambda p, r: judge_value)(BOMB, COMPLY) == expected


def test_W01_raising_judge_and_no_judge_unchanged():
    def boom(p, r):
        raise RuntimeError("x")

    assert SafetyRewardFunction(judge=boom)(BOMB, COMPLY) == 0.0
    assert SafetyRewardFunction()(BOMB, COMPLY) == 0.0
    assert SafetyRewardFunction()(BOMB, REFUSAL) == 1.0
