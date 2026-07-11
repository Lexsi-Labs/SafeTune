"""Optional unified YAML config (legacy API - prefer per-Trainer Configs)."""
from __future__ import annotations

try:
    from safetune.core.safety_config import (
        EvalSafetyConfig,
        GuardrailConfig,
        InferenceSafetyConfig,
        TrainingSafetyConfig,
        UnifiedSafetyConfig,
    )
except ImportError as e:  # pragma: no cover - defensive
    raise ImportError(f"UnifiedSafetyConfig unavailable: {e}") from e

__all__ = [
    "UnifiedSafetyConfig",
    "TrainingSafetyConfig",
    "InferenceSafetyConfig",
    "GuardrailConfig",
    "EvalSafetyConfig",
]
