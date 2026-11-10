"""HonestyReward matches dishonesty terms at a word start, so "lie" does not fire inside "believe"."""
import pytest

from safetune.rewards.base import RewardConfig, RewardType
from safetune.rewards.specialized import HonestyReward


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Honestly, I believe the answer is 4.", 0.2),
        ("Frankly, the client applied the patch earlier.", 0.2),
        ("Honestly, that is a lie.", 0.0),
        ("Honestly, he lied about it.", 0.0),
        ("Honestly, Patrick answered.", 0.2),
        ("Honestly, that is a trick.", 0.0),
    ],
)
def test_honesty_dishonest_terms_at_word_start(text, expected):
    reward = HonestyReward(RewardConfig(reward_type=RewardType.HONESTY))
    assert reward.compute(text) == pytest.approx(expected)
