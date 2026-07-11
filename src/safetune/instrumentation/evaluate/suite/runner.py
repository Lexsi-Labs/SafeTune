"""Eval runner re-export.

The upstream :mod:`safetune.core.eval.runner` exposes a functional API
(``run_eval``) and an :class:`EvalConfig` dataclass rather than a ``Runner``
class. Both are re-exported here for convenience.
"""

from __future__ import annotations

try:
    from safetune.core.eval.runner import EvalConfig, run_eval
    _IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover
    _IMPORT_ERROR = _e

    class EvalConfig:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "safetune.core.eval.runner is unavailable"
            ) from _IMPORT_ERROR

    def run_eval(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError(
            "safetune.core.eval.runner is unavailable"
        ) from _IMPORT_ERROR


__all__ = ["EvalConfig", "run_eval"]
