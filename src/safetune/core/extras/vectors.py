"""Steering vector extraction utilities."""
from __future__ import annotations

try:
    from safetune.core.runtime.inference.vector_extraction import (
        SteeringVectorExtractor,
        VectorExtractionConfig,
    )
except ImportError as e:  # pragma: no cover - defensive
    raise ImportError(f"SteeringVectorExtractor unavailable: {e}") from e

__all__ = ["SteeringVectorExtractor", "VectorExtractionConfig"]
