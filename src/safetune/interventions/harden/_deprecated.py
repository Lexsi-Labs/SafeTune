"""Old names of the harden ``transformers.Trainer`` subclasses, kept for one release.

``safetune.harden.LisaTrainer`` is now the high-level trainer (the same class as
``safetune.runner.harden.LisaTrainer``). The ``transformers.Trainer`` subclass it
runs is ``LisaHFTrainer``. Importing the old name from a submodule
(``safetune.harden.lisa.LisaTrainer``) still returns that subclass, with a
``DeprecationWarning``. Remove in 0.3.
"""
from __future__ import annotations

import sys
import warnings


def renamed(module: str, **old_to_new: str):
    """A module ``__getattr__`` that serves each old name as its new one, with a warning."""
    def __getattr__(name: str):
        if name not in old_to_new:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        new = old_to_new[name]
        warnings.warn(
            f"{module}.{name} is now {new}; the old name stops working in 0.3. "
            f"safetune.harden.{name} is the high-level trainer "
            f"({name}(model, tokenizer).train(dataset)).",
            DeprecationWarning, stacklevel=2)
        return getattr(sys.modules[module], new)
    return __getattr__
