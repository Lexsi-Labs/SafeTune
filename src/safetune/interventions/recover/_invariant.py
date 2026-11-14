"""Defense-in-depth: assert every Recover entrypoint mutates model weights.

Background: three Recover methods in the planning CSV (PKE, NLSR, SafeReact)
once produced "Same as base!?" rows, which an end-user reads as "the method
ran but did nothing." The empirical diagnostic at
``tests/diagnostic/test_recover_state_dict_mutation.py`` catches the pattern
at test time; this decorator catches it at runtime, where it is otherwise
silent.

Usage::

    from ._invariant import assert_mutates

    @assert_mutates("apply_pke")
    def apply_pke(model, ...):
        ...

The decorator computes a cheap GLOBAL weight signature (sum + sum-of-squares
over every parameter) before the call, lets the wrapped function run untouched,
then logs a WARNING (not an exception, so optional pipelines do not break) if
the signature is unchanged afterward. A global signature changes whenever *any*
weight changes, so — unlike sampling a handful of params — it does not raise
false alarms on methods that deliberately skip embeddings / norms / biases.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _mutation_signature(model: Any) -> Optional[float]:
    """Cheap global fingerprint of all parameter weights (no per-param clones).

    Returns ``sum(w) + sum(w**2)`` accumulated over every parameter, or ``None``
    if it can't be computed. The squared term guards against sign-cancelling
    edits that would leave the plain sum unchanged.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - defensive
        return None
    try:
        s = 0.0
        sq = 0.0
        with torch.no_grad():
            for p in model.parameters():
                if p is None:
                    continue
                pf = p.detach().float()
                s += pf.sum().item()
                sq += pf.pow(2).sum().item()
        return s + sq
    except Exception:  # pragma: no cover - defensive
        return None


def require_matched(method: str, n_matched: int, model: Any, reference: Any) -> None:
    """Raise when no parameter of ``model`` has a same-named counterpart in ``reference``.

    Name-matched weight arithmetic skips every name it cannot find in the reference, so a
    PEFT-wrapped model against a bare one (``base_model.model.`` prefix, LoRA factors) or any
    other renamed pair used to return the model untouched. ``reference`` is a module or a
    state dict. Count matches before any ``param_filter`` / shape / dimension skip.
    """
    if n_matched:
        return

    def sample(x: Any) -> list:
        return list((x.state_dict() if hasattr(x, "state_dict") else x).keys())[:2]

    raise ValueError(
        f"{method}: no parameter name of the model is in the reference weights, so nothing would "
        f"change (model has {sample(model)}, reference has {sample(reference)}). Both must share "
        "parameter names; merge a PEFT adapter into its base first with merge_and_unload()."
    )


def assert_mutates(method_name: str) -> Callable:
    """Decorator that logs a WARNING if the model's weights are unchanged.

    Args:
        method_name: human-readable label used in the log message (e.g.
            ``"apply_pke"``).
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(model: Any, *args: Any, **kwargs: Any) -> Any:
            before = _mutation_signature(model)
            result = func(model, *args, **kwargs)
            # The wrapped method may either mutate `model` in place or return a
            # new module. Inspect whichever object the caller will use next.
            target = result if hasattr(result, "parameters") else model
            after = _mutation_signature(target)

            if (
                before is not None
                and after is not None
                and abs(after - before) < 1e-6
            ):
                logger.warning(
                    "%s: model weights appear unchanged after apply "
                    "(global signature delta = %.2e). The method may have "
                    "silently no-op'd. Verify that your reference inputs differ "
                    "from the target model.",
                    method_name,
                    abs(after - before),
                )
            return result

        return wrapper

    return decorator


__all__ = ["assert_mutates", "require_matched"]
