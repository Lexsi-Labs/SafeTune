"""W05 LogicalConsistencyReward: the ("all", "none") contradiction pair must match whole words.

reward-changing. "all" matched inside "ball"/"overall"/"install" and "none" inside "nonetheless",
so ordinary text lost 0.2. The ("always", "never") and ("true", "false") pairs are unchanged
(not part of this approval).
"""
import pytest

from safetune.rewards.base import RewardConfig, RewardType
from safetune.rewards.math import LogicalConsistencyReward

R = LogicalConsistencyReward(RewardConfig(reward_type=RewardType.LOGICAL_CONSISTENCY, params={}))

# BEFORE-fix scores of inputs that were already correct; must stay bit-identical.
UNCHANGED = {
    "All birds fly. None of them swim. It is so.": 0.8,
    "All good. Fine. Ok.": 1.0,
    "none here. all there. ok": 0.8,
    "small. tall. wall.": 1.0,
    "Always do this. Never do that. Thus done.": 0.9,
    "It is true. It is false. Therefore, because.": 1.0,
    "One. Two.": 1.0,
    "All cats are mammals. None are fish. Thus, true.": 0.9,
    "ALL men. NONE left. y": 0.8,
    "all-in. none-the-less. z": 0.8,
    "Overall it is fine. Nobody. Done.": 1.0,
    "All, none. q. r": 0.8,
}


@pytest.mark.parametrize("text", list(UNCHANGED))
def test_W05_all_none_unchanged_inputs_bit_identical(text):
    assert R.compute(text) == UNCHANGED[text]


def test_W05_all_none_inside_words_not_a_contradiction():
    # before: 0.8 ("all" in "ball", "none" in "nonetheless")
    assert R.compute("The ball is round. Nonetheless it holds. Overall fine.") == 1.0
