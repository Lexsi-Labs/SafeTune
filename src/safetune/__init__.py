"""
SafeTune — a library of LLM-safety methods.

SafeTune is **a library of safety methods, not a pipeline.** Each
task has many independent methods that solve it by *different mechanisms* — you
pick one per task, you do not run them in sequence. Methods are organized into a
2-tier, input-keyed taxonomy (the single source of truth is
``docs/getting-started/taxonomy.md``):

    TIER 1 · INTERVENTIONS — methods that CHANGE a model's safety, keyed by
                             what you input / when safety is enforced:
        Train-time      input: base model + your FT data    -> harden
        Weight-space    input: a finished / drifted model    -> recover + unlearn
        Inference-time  input: any model + steering artifacts -> steer

    TIER 2 · INSTRUMENTATION — methods that OBSERVE safety; they serve the
                               interventions and are usable standalone:
        Diagnose  -> interpret  (locate safety: directions, neurons, circuits)
        Measure   -> evaluate   (redteam stressors + benchmark/judge eval)

Each Tier-1 box is a *catalog of independent alternatives* — each member is a
different mechanism, not a step. The Measure pillar is named ``evaluate`` (the
old name ``verify`` oversold it — it measures, it cannot verify).

SafeTune ships many methods; they are NOT all equal. A faithfulness audit
labels every method against its cited paper — see ``docs/trust/feature-map.md`` and
``docs/trust/scope.md``. Only methods marked ✅ faithfully implement their paper; methods
marked 🟠/🔴/⚫ run but must not be cited as the named method. Check the badge
before relying on a method.

Quickstart:
    >>> import safetune
    >>> # Weight-space — post-hoc weight patching (Recover / Unlearn)
    >>> model = safetune.recover.task_arithmetic(model, base=base, aligned=aligned)
    >>> # Train-time — a Harden trainer replaces your SFT trainer
    >>> trainer = safetune.harden.SafeGradTrainer(model, ...)
    >>> # Inference-time — wrap a frozen model
    >>> model = safetune.steer.RefusalDirectionModel(model, direction=vec)
    >>> # Measure — score a model
    >>> results = safetune.evaluate.evaluate(model, benchmarks=["harmbench"])
"""

__version__ = "1.0.0"
__author__ = "Lexsi Labs"

# ── Backend guards ──────────────────────────────────────────────────────────
# SafeTune is a PyTorch-only toolkit. Disable the TensorFlow / Flax backends
# in `transformers` *before* it is imported by any submodule. Otherwise
# `from transformers import Trainer` triggers transformers' TF integration
# path, which hard-fails on environments shipping Keras 3 without the
# `tf-keras` shim — an unrelated transitive import error that would otherwise
# make every HF-Trainer-based HARDEN method look broken.
import os as _os

_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# ── Tier 1 · Interventions ──────────────────────────────────────────────────
from .interventions import recover, harden, steer, unlearn

# ── Tier 2 · Instrumentation ────────────────────────────────────────────────
from .instrumentation import evaluate, interpret

# Register pillar packages so `from safetune.harden import ...` resolves
# (the physical directories live under interventions/ and instrumentation/).
import sys as _sys
_sys.modules.setdefault("safetune.recover", recover)
_sys.modules.setdefault("safetune.harden", harden)
_sys.modules.setdefault("safetune.steer", steer)
_sys.modules.setdefault("safetune.unlearn", unlearn)
_sys.modules.setdefault("safetune.interpret", interpret)
_sys.modules.setdefault("safetune.evaluate", evaluate)

# `safetune.verify` was removed — use `safetune.evaluate`.

from . import core

# ── Supporting namespaces ───────────────────────────────────────────────────
from . import data
from . import rewards
from . import utils

# NOTE: `pipelines.pipeline` and `training.orchestrator.run_sft/dpo/ppo/grpo`
# were removed from the public surface — they dispatched to undefined classes /
# a missing `core.backend_factory`. The modules remain on disk but are not
# advertised; use the pillar entry points (`harden.*Trainer`, `recover.apply_*`,
# etc.) directly. See docs/getting-started/taxonomy.md.

__all__ = [
    # Tier 1 · interventions
    "recover",
    "unlearn",
    "harden",
    "steer",
    # Tier 2 · instrumentation
    "interpret",
    "evaluate",
    # supporting
    "core",
    "data",
    "rewards",
    "utils",
]
