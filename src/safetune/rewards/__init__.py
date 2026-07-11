"""
Reward functions for RL training.

This module provides comprehensive reward functions for reinforcement learning
from human feedback (RLHF) training, covering all types of tasks and scenarios.
"""

from .base import RewardType, RewardConfig, RewardFunction, CompositeReward
from .factory import RewardFunctionFactory
from .text import LengthReward, CoherenceReward
from .nlp import SentimentReward, BLEUReward, ROUGEReward
from .safety import SafetyReward, ToxicityReward
from .math import MathCorrectnessReward
from .code import CodeSyntaxReward

# Import training classes from TRL backend (re-export for backward compatibility)
try:
    from ..backends.trl.rewards import (
        RewardModelTrainer,
        RewardModelDataset,
        RewardModelValidator,
        RewardModelLoader,
    )
except ImportError:
    # Fallback to local training module if TRL backend not available
    from .training import (
        RewardModelTrainer,
        RewardModelDataset,
        RewardModelValidator,
        RewardModelLoader,
    )

# Import factory module
from . import factory

# Register default reward functions
from . import registry
from .registry import RewardRegistry

# ---------------------------------------------------------------------------
# Register SafetyRewardFunction as a custom reward callable.
# Name "aligned_safety" is used to distinguish from the model-based "safety"
# entry already registered by _initialize_default_rewards().
# ---------------------------------------------------------------------------
try:
    from .safety import SafetyRewardFunction as _SafetyRewardFunction

    def _build_aligned_safety(config: RewardConfig) -> _SafetyRewardFunction:
        """Factory: build a SafetyRewardFunction from a RewardConfig."""
        weight = getattr(config, "weight", 1.0)
        params = getattr(config, "params", {}) or {}
        return _SafetyRewardFunction(
            harmfulness_weight=float(params.get("harmfulness_weight", weight)),
            over_refusal_penalty=float(params.get("over_refusal_penalty", 0.3)),
        )

    RewardRegistry.register_custom_reward(
        "aligned_safety",
        _build_aligned_safety,
        config=RewardConfig(
            reward_type=RewardType.SAFETY,
            weight=1.0,
            params={"harmfulness_weight": 1.0, "over_refusal_penalty": 0.3},
        ),
    )

    from .safety import SafetyRewardFunction  # re-export
except Exception:
    pass  # rewards.safety may be unavailable in minimal installs

__all__ = [
    "RewardType",
    "RewardConfig",
    "RewardFunction",
    "RewardFunctionFactory",
    "CompositeReward",
    "LengthReward",
    "CoherenceReward",
    "SentimentReward",
    "SafetyReward",
    "SafetyRewardFunction",
    "BLEUReward",
    "ROUGEReward",
    "MathCorrectnessReward",
    "CodeSyntaxReward",
    "ToxicityReward",
    "RewardModelTrainer",
    "RewardModelDataset",
    "RewardModelValidator",
    "RewardModelLoader",
    "factory",
    "registry",
    "RewardRegistry",
]
