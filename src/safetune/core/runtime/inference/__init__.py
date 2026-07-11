"""
Inference-time Safety Extensions
Provides Dynamic Activation Patching and LLM Safeguard (predict-before-generate).
"""

from .dynamic_patching import DynamicPatchingConfig, HookedGenerationWrapper, PatchingStrategy
from .safeguard import LLMSafeguardPredictor

__all__ = [
    "DynamicPatchingConfig",
    "HookedGenerationWrapper",
    "PatchingStrategy",
    "LLMSafeguardPredictor",
]
