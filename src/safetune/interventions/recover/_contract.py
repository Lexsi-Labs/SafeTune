"""Uniform model-input contract for the RECOVER pillar.

Every RECOVER entry point edits one *target* model — the finished / drifted
checkpoint you want to repair — optionally guided by reference models
(``base`` = pre-alignment, ``aligned`` = the safe reference). Historically the
target argument was named inconsistently across methods (``model`` /
``finetuned`` / ``target``).

:func:`accept_target_alias` is a thin, fully-additive decorator that lets every
``apply_*`` accept the **canonical** keyword ``target=`` regardless of how its
first parameter is spelled, while the legacy names (``model=`` / ``finetuned=``)
keep working. Positional calls are unaffected.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

# All accepted spellings of the primary (target) model argument.
_PRIMARY_ALIASES = ("target", "finetuned", "model")


def accept_target_alias(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``fn`` so its primary model argument accepts any of
    ``target=`` / ``finetuned=`` / ``model=`` as a keyword.

    The canonical name is ``target=``; the others are back-compat aliases.
    """
    # Use the *literal* signature (follow_wrapped=False): some RECOVER methods
    # are already wrapped (e.g. ``@assert_mutates``) and their real first
    # parameter — the one a call must satisfy — is the wrapper's, not the
    # inner function's.
    try:
        params = list(
            inspect.signature(fn, follow_wrapped=False).parameters.values()
        )
    except (TypeError, ValueError):  # pragma: no cover - builtins etc.
        return fn
    positional = [
        p.name for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    first = positional[0] if positional else None
    if first is None:
        return fn

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Only remap when the real first parameter was not already supplied.
        if first not in kwargs:
            for alias in _PRIMARY_ALIASES:
                if alias != first and alias in kwargs:
                    kwargs[first] = kwargs.pop(alias)
                    break
        return fn(*args, **kwargs)

    return wrapper


__all__ = ["accept_target_alias"]
