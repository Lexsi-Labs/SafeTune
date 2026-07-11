"""PTST evaluator re-export."""

from __future__ import annotations

try:
    from safetune.core.eval.ptst import PTSTConfig, PTSTEvaluator
    _IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover
    _IMPORT_ERROR = _e

    class PTSTEvaluator:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "safetune.core.eval.ptst is unavailable"
            ) from _IMPORT_ERROR

    class PTSTConfig:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "safetune.core.eval.ptst is unavailable"
            ) from _IMPORT_ERROR


__all__ = ["PTSTConfig", "PTSTEvaluator"]
