"""Judge adapter re-exports.

Thin re-export of the judge-adapter classes whose single source of truth is
:mod:`safetune.core.eval.judge_adapters`. This module adds only the friendly
``verify``-surface aliases (``LocalClassifierJudge`` / ``APIJudge``) and a
graceful ImportError stub; it does **not** redefine any adapter.
"""

from __future__ import annotations

try:
    from safetune.core.eval.judge_adapters import (
        APIJudgeAdapter,
        BatchedClassifierJudgeAdapter,
        ClassifierJudgeAdapter,
        JudgeAdapter,
        LocalJudgeAdapter,
        ScorerJudgeAdapter,
        build_judge_adapter,
    )
    _IMPORT_ERROR = None

    # Friendly aliases for the verify surface.
    LocalClassifierJudge = ClassifierJudgeAdapter
    APIJudge = APIJudgeAdapter
except Exception as _e:  # pragma: no cover
    _IMPORT_ERROR = _e

    class JudgeAdapter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "safetune.core.eval.judge_adapters is unavailable"
            ) from _IMPORT_ERROR

    class LocalClassifierJudge:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "safetune.core.eval.judge_adapters is unavailable"
            ) from _IMPORT_ERROR

    class APIJudge:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "safetune.core.eval.judge_adapters is unavailable"
            ) from _IMPORT_ERROR

    LocalJudgeAdapter = LocalClassifierJudge  # type: ignore[assignment]
    ClassifierJudgeAdapter = LocalClassifierJudge  # type: ignore[assignment]
    BatchedClassifierJudgeAdapter = LocalClassifierJudge  # type: ignore[assignment]
    APIJudgeAdapter = APIJudge  # type: ignore[assignment]
    ScorerJudgeAdapter = LocalClassifierJudge  # type: ignore[assignment]
    build_judge_adapter = None  # type: ignore[assignment]


__all__ = [
    "JudgeAdapter",
    "LocalJudgeAdapter",
    "ClassifierJudgeAdapter",
    "BatchedClassifierJudgeAdapter",
    "APIJudgeAdapter",
    "ScorerJudgeAdapter",
    "LocalClassifierJudge",
    "APIJudge",
    "build_judge_adapter",
]
