"""Evaluate pillar — Tier 2 Instrumentation, the *Measure* function.

(Renamed from ``verify``: the old name oversold it — this pillar *measures*
safety, it cannot *verify* it.)

Two distinct functions, kept cleanly separate:

- :mod:`safetune.evaluate.redteam` — stressors. Only the audit-verified ones:
  ``AbliterationAttack`` (weight-space drift condition) and ``BoNAttack``.
- :mod:`safetune.evaluate.suite` — judges, benchmarks, metrics, and the
  ``SpectralEntropyMonitor`` drift monitor; a thin wrapper over external
  harnesses (TamperBench, ``cthetha-eval``).

The 12 broken / approximate in-house attack reimplementations were removed —
see ``docs/getting-started/taxonomy.md`` and ``docs/trust/feature-map.md``.
"""

from . import redteam, suite
from .redteam import AbliterationAttack, BoNAttack, BoNConfig, AUGMENTATIONS
from .suite import (
    evaluate,
    TamperBenchEvaluator,
    JudgeAdapter,
    SpectralEntropyMonitor,
    SpectralMonitorConfig,
)

__all__ = [
    "redteam",
    "suite",
    # red-team surface
    "AbliterationAttack",
    "BoNAttack", "BoNConfig", "AUGMENTATIONS",
    # evaluation surface
    "evaluate",
    "TamperBenchEvaluator",
    "JudgeAdapter",
    "SpectralEntropyMonitor",
    "SpectralMonitorConfig",
]
