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

import copy
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


def require_matched(
    method: str, model: Any, *references: Any, strict: bool = False, sparse: bool = False
) -> None:
    """Check that ``model`` and the reference weights share parameter names, before any edit.

    Name-matched weight arithmetic skips every name it cannot find in the reference, so a
    PEFT-wrapped model against a bare one (``base_model.model.`` prefix, LoRA factors) or any
    other renamed pair used to return the model untouched. Each reference is a module or a
    name-keyed mapping (a state dict); a name must be in all of them.

    * No name in common: always raises ``ValueError``.
    * Partial overlap: logs one summary line with the counts (a warning when something is
      left out), and raises ``ValueError`` when ``strict`` is true.

    What partial coverage means depends on the method. By default it is a model parameter that
    has no same-named entry in the references. ``sparse=True`` is for a reference that
    names only the weights to edit (a neuron map): it is a reference name that is not a
    model parameter. Filters, shape and dimension skips inside a method are never counted.
    """
    def names(x: Any) -> set:
        return set((x.state_dict() if hasattr(x, "state_dict") else x).keys())

    own = [n for n, _ in model.named_parameters()]
    shared = set.intersection(*(names(r) for r in references))
    if sparse:
        total, missing = len(shared), sorted(shared - set(own))
        matched = total - len(missing)
        side = "reference entries that are not model parameters"
    else:
        missing = [n for n in own if n not in shared]
        total, matched = len(own), len(own) - len(missing)
        side = "model parameters that are not in the reference"
    if not matched:
        def sample(x: Any) -> list:
            return sorted(names(x))[:2]

        raise ValueError(
            f"{method}: no parameter name of the model is in the reference weights, so nothing "
            f"would change (model has {sorted(own)[:2]}, reference has {sample(references[0])}). "
            "Both must share parameter names; merge a PEFT adapter into its base first with "
            "merge_and_unload() (apply_wise_ft and apply_prepost_merge do it with merge_adapter=True)."
        )
    summary = f"{method}: {matched} of {total} matched by name, {len(missing)} {side}"
    if missing and strict:
        raise ValueError(f"{summary} (e.g. {missing[:3]}); strict=True requires every one to match.")
    (logger.warning if missing else logger.info)(
        "%s%s.", summary, f" (e.g. {missing[:3]}), left unchanged" if missing else ""
    )


def merged_adapter_copy(model: Any, method: str) -> Any:
    """A bare copy of a PEFT ``model`` with its adapter merged in; ``model`` is left as it is."""
    if not hasattr(model, "merge_and_unload"):
        raise ValueError(f"{method}: merge_adapter=True needs a PEFT model with an adapter to merge.")
    return copy.deepcopy(model).merge_and_unload()


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


__all__ = ["assert_mutates", "merged_adapter_copy", "require_matched"]
