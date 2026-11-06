"""TemporalConsistencyReward matches temporal pairs as whole words: "past" is not in "pasta"."""
import pytest

from safetune.rewards.base import RewardConfig, RewardType
from safetune.rewards.specialized import TemporalConsistencyReward


@pytest.mark.parametrize(
    "previous, text, expected",
    [
        ("We ate pasta.", "See you in the future.", 1.0),
        ("We met before lunch.", "See you this afternoon.", 1.0),
        ("That was the past.", "This is the future.", 0.7),
        ("Yesterday was fine.", "Tomorrow is better.", 0.7),
    ],
)
def test_temporal_pairs_match_whole_words(previous, text, expected):
    reward = TemporalConsistencyReward(RewardConfig(reward_type=RewardType.TEMPORAL_CONSISTENCY))
    assert reward.compute(text, previous_texts=[previous]) == pytest.approx(expected)
