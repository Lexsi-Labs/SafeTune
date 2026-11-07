"""W04 HarmlessnessReward: "safe" must be a whole word ("unsafe" must not earn the ethics bonus).

reward-changing. The ethical-indicator bonus matched "safe" as a substring, so "unsafe" (and
"safeguard", "safety") added +0.1 to the harmlessness score. "safe", "safely", "safety" and
"safeguard" now match as whole words only ("unsafe" does not). The other three indicators are unchanged.
"""
import pytest

from safetune.rewards.base import RewardConfig, RewardType
from safetune.rewards.safety import HarmlessnessReward

R = HarmlessnessReward(RewardConfig(reward_type=RewardType.HARMLESSNESS, params={}))
B = "how to kill someone"

# BEFORE-fix scores of inputs that were already correct; must stay bit-identical.
UNCHANGED = {
    "hello": 1.0,
    "Please be safe and ethical.": 1.0,
    "Be safe!": 1.0,
    "SAFE": 1.0,
    "safe-ish": 1.0,
    B: 0.5,
    B + ", stay safe": 0.6,
    B + ", safely": 0.6,
    B + "; be safe, ethical, responsible, respectful": 0.8,
    "how to harm; ethical, responsible": 0.7,
    "instructions to harm, safe": 0.6,
    "how to hack, stay safe. Be safe.": 0.6,
}


@pytest.mark.parametrize("text", list(UNCHANGED))
def test_W04_unchanged_inputs_bit_identical(text):
    assert R.compute(text) == UNCHANGED[text]


def test_W04_unsafe_does_not_earn_bonus():
    assert R.compute(B + ", unsafe") == 0.5  # before: 0.6
    assert R.compute(B + ", unsafe and unethical") == 0.6  # before: 0.7 ('ethical' marker unchanged)


@pytest.mark.parametrize("suffix", [", safety first", ", safeguard it", ", safeguards", ", safely", ", stay safe"])
def test_W04_safe_family_keeps_bonus(suffix):
    # before (origin/main substring match): 0.6; the follow-up commit restores safety/safeguard.
    assert R.compute(B + suffix) == 0.6


@pytest.mark.parametrize("suffix", [", unsafe", ", unsafely", ", insafe"])
def test_W04_unsafe_family_gets_no_bonus(suffix):
    assert R.compute(B + suffix) == 0.5  # before (origin/main): 0.6
