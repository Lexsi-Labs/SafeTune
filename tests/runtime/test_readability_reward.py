"""ReadabilityReward caps each part at 1, so a very short sentence cannot hide long words or punctuation."""
import pytest

from safetune.rewards.base import RewardConfig, RewardType
from safetune.rewards.text import ReadabilityReward


@pytest.mark.parametrize(
    "text, expected",
    [
        ("It is: so; we go.", 0.9),
        ("It is so we go.", 1.0),
        ("Supercalifragilistic. Antidisestablishmentarianism.", 0.5),
        (" ".join(["abcde"] * 40), 0.5 * ((1 - 25 / 30) + 0.9)),
    ],
)
def test_readability_parts_are_capped(text, expected):
    reward = ReadabilityReward(RewardConfig(reward_type=RewardType.READABILITY))
    assert reward.compute(text) == pytest.approx(expected)
