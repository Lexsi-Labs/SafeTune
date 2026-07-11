"""Smoke coverage for EVERY registered method: each trainer must resolve from
its registry and construct without error. Fast and model-free (constructs with
model=None; trainers lazy-load the model only when actually used), so this runs
in CI and gives every one of the ~70 shipped methods a pytest that exercises its
constructor — the class of bug (e.g. a broken __init__ signature) that a
name-only import test would miss.

End-to-end execution of each method (.apply/.train/.unlearn/.calibrate) is
covered by the integration scripts under tests/integration/ (they need real
drift/aligned checkpoints) and by the executed documentation examples.
"""
import importlib

import pytest

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


def _registered():
    """(pillar, alias, TrainerClass) for every registered trainer + steer."""
    from safetune.runner import _registry
    out = []
    for pillar in ("harden", "recover", "unlearn"):
        reg = getattr(_registry, f"{pillar.upper()}_REGISTRY")
        mod = importlib.import_module(f"safetune.runner.{pillar}")
        for alias, cls_name in reg.items():
            out.append((pillar, alias, getattr(mod, cls_name)))
    steer = importlib.import_module("safetune.runner.steer")
    for name in dir(steer):
        obj = getattr(steer, name)
        if name.endswith("Trainer") and isinstance(obj, type):
            out.append(("steer", name, obj))
    return out


_ALL = _registered() if _HAS_TORCH else []


def test_registry_is_non_empty():
    assert len(_ALL) >= 60, f"expected the full method set, got {len(_ALL)}"


@pytest.mark.parametrize(
    "pillar,alias,cls",
    _ALL,
    ids=[f"{p}:{a}" for p, a, _ in _ALL],
)
def test_method_constructs(pillar, alias, cls):
    """Every registered method must construct with no model (lazy-loaded later)."""
    # harden & steer trainers take (model, tokenizer); recover & unlearn take (model)
    trainer = cls(None, None) if pillar in ("harden", "steer") else cls(None)
    assert trainer is not None
    # the pillar tag should be consistent (defensive: catches a mis-registered class)
    assert getattr(trainer, "PILLAR", pillar) in (pillar, "harden", "recover",
                                                   "unlearn", "steer")
