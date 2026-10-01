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
import os
import warnings
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


_HERE = os.path.dirname(os.path.abspath(__file__))


def keyword_refs(*legacy: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Accept the old positional form of a function whose arguments after the
    first are now keyword-only, with a ``DeprecationWarning``, for one release.

    ``legacy`` is the function's old positional order after the target model.
    Recover functions did not share one (``task_arithmetic(ft, base, aligned)``
    but ``somf_merge(ft, aligned, base)``), so switching methods with positional
    arguments swapped ``base`` and ``aligned`` silently. Put this decorator
    directly on the ``def``, under ``@assert_mutates``.
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if len(args) < 2:
                return fn(*args, **kwargs)
            names = legacy[:len(args) - 1]
            warnings.warn(
                f"{fn.__name__}: passing {', '.join(names)} by position is deprecated "
                f"and stops working in 0.3. Use keywords: "
                f"{fn.__name__}(model, {', '.join(n + '=...' for n in names)}).",
                DeprecationWarning, skip_file_prefixes=(_HERE,))
            # Two sources for one name raise TypeError, as the old call did.
            return fn(args[0], *args[1 + len(names):], **dict(zip(names, args[1:])), **kwargs)
        return wrapper
    return decorator


__all__ = ["accept_target_alias", "keyword_refs"]
