"""
Metrics registry for evaluation.
"""

from .base import Metric
from .generic import PerplexityMetric, AccuracyMetric
from .text import BleuMetric, RougeMetric
from .rl import KLDivergenceMetric, RewardAccuracyMetric, PolicyEntropyMetric
from .math import MathAccuracyMetric
from .code import PassAtKMetric
from .safety import (
    HarmfulnessMetric,
    OverRefusalMetric,
    CapabilityRegressionMetric,
    JailbreakSuccessMetric,
    JudgeBackendOutput,
    score_with_backend,
    score_with_backend_checked,
    compute_safety_gates,
    compute_artifact_promotion_gate,
    compute_safety_suite,
)
__all__ = [
    "Metric",
    "PerplexityMetric",
    "AccuracyMetric",
    "BleuMetric",
    "RougeMetric",
    "KLDivergenceMetric",
    "RewardAccuracyMetric",
    "PolicyEntropyMetric",
    "MathAccuracyMetric",
    "PassAtKMetric",
    "HarmfulnessMetric",
    "OverRefusalMetric",
    "CapabilityRegressionMetric",
    "JailbreakSuccessMetric",
    "JudgeBackendOutput",
    "score_with_backend",
    "score_with_backend_checked",
    "compute_safety_gates",
    "compute_artifact_promotion_gate",
    "compute_safety_suite",
]
