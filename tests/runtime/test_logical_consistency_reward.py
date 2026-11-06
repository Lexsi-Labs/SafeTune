"""LogicalConsistencyReward matches contradiction pairs as whole words: "all" is not in "small"."""
import pytest

from safetune.rewards.base import RewardConfig, RewardType
from safetune.rewards.math import LogicalConsistencyReward


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The small box is here. Nonetheless it fits.", 1.0),
        ("We construed it well. The falsehood is gone.", 1.0),
        ("It is always true. It is never false.", 0.6),
        ("All of it fits. None of it breaks.", 0.8),
    ],
)
def test_logical_consistency_pairs_match_whole_words(text, expected):
    reward = LogicalConsistencyReward(RewardConfig(reward_type=RewardType.LOGICAL_CONSISTENCY))
    assert reward.compute(text) == pytest.approx(expected)
