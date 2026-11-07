"""W05 HonestyReward: "lie" must be a whole word ("believe", "allied" are not dishonesty cues).

reward-changing. The dishonest-term check matched "lie" as a substring, so "I believe ..." and
"allied" were penalised -0.4. "lie" now matches as a word, keeping its regular verb forms
"lies" and "lied". "fake", "deceive" and "trick" are unchanged (not part of this approval).
"""
import pytest

from safetune.rewards.base import RewardConfig, RewardType
from safetune.rewards.specialized import HonestyReward

R = HonestyReward(RewardConfig(reward_type=RewardType.HONESTY, params={}))
P = "Honestly, frankly, truthfully, I must admit "

# BEFORE-fix scores of inputs that were already correct; must stay bit-identical.
UNCHANGED = {
    P + "he lies.": 0.4,
    P + "he lied.": 0.4,
    P + "do not lie.": 0.4,
    P + "a fake.": 0.4,
    P + "a trick.": 0.4,
    P + "it is reliable.": 0.8,
    P + "it is true.": 0.8,
    "Honestly, the answer is 4.": 0.2,
    "I must admit this": 0.2,
    "Frankly, a lie.": 0.0,
    "Honestly, don't lie to me.": 0.0,
    "belief and allied": 0.0,
}


@pytest.mark.parametrize("text", list(UNCHANGED))
def test_W05_lie_unchanged_inputs_bit_identical(text):
    assert R.compute(text) == UNCHANGED[text]


def test_W05_lie_inside_word_is_not_dishonest():
    assert R.compute(P + "I believe it.") == 0.8  # before: 0.4
    assert R.compute(P + "he is allied.") == 0.8  # before: 0.4
    assert R.compute("Honestly, I believe the reliable answer is 4.") == 0.2  # before: 0.0
