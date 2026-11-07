"""One set of steering hooks per model and method.

Two wrappers of the same method on one model would steer it twice, so installing a
wrapper removes the hooks an earlier one left. That earlier wrapper is told: its hook
list is emptied and a warning names it, so it cannot go on reporting unsteered results
as steered ones without anyone noticing.
"""
from __future__ import annotations

import logging
import weakref

logger = logging.getLogger(__name__)

# model -> weakref to the wrapper that owns its hooks, per method. Kept off the model so
# pickling and deepcopy of the model do not carry it.
_OWNERS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def release_previous(wrapper, key: str, hooks_attr: str = "_hooks") -> None:
    """Remove the hooks an earlier ``key`` wrapper left on ``wrapper.model``."""
    model = wrapper.model
    for h in getattr(model, f"_safetune_{key}_handles", None) or []:
        h.remove()
    ref = _OWNERS.get(model, {}).get(key)
    prev = ref() if ref is not None else None
    if prev is not None and prev is not wrapper and getattr(prev, hooks_attr, None):
        logger.warning("%s: another %s wrapper was installed on this model; its hooks are removed "
                       "and it no longer steers.", key, type(prev).__name__)
        getattr(prev, hooks_attr).clear()


def claim(wrapper, key: str, handles) -> None:
    """Record ``wrapper`` as the owner of ``handles`` on its model."""
    try:
        setattr(wrapper.model, f"_safetune_{key}_handles", list(handles))
        _OWNERS.setdefault(wrapper.model, {})[key] = weakref.ref(wrapper)
    except (AttributeError, TypeError):  # a model that refuses new attributes or is not weak-referenceable
        pass
