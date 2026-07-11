"""
Safety evaluation task registry and helpers.

This module provides a small registry for *safety-focused* evaluation tasks
built on top of SafeTune's Safety Packs and pack runners. It is
inspired by DeepSafe's registry/config-driven design but implemented
entirely inside this library.

Key concepts:
- SafetyEvalTask: declarative config for running a safety pack (e.g. HarmBench)
  with a given judge backend, metric set, and thresholds.
- SafetyEvalRegistry: in-memory registry for tasks, with helpers to create
  default HarmBench / JailbreakBench / XSTest / HH-RLHF tasks.
- run_safety_eval_from_hf: convenience entrypoint that auto-loads a pack
  from Hugging Face and executes the corresponding pack runner.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.data_compiler.safety_packs import SafetyPack, resolve_pack
from ..core.data_compiler.pack_runners import (
    PackRunResult,
    load_pack_from_hf,
    run_pack,
)
from .registry import EvalRegistry


@dataclass
class SafetyEvalTask:
    """Configuration for a single safety evaluation task.

    Attributes:
        name: Logical task name (e.g. "harmbench_default").
        pack_name: Safety pack identifier ("harmbench", "jailbreakbench", "xstest", "hh_rlhf", ...).
        pack_version: Optional version string; if empty, uses the default in SafetyPack.
        judge_backend: High-level backend selection ("local", "api", "classifier").
        judge_params: Backend-specific parameters (e.g. model names, endpoints).
        metrics: Metric names to compute; defaults to the safety suite.
        thresholds: Gate thresholds (e.g. {"harmfulness_max": 0.1}).
    """

    name: str
    pack_name: str
    pack_version: str = ""
    judge_backend: str = "local"
    judge_params: Dict[str, Any] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=lambda: EvalRegistry.get_safety_suite_metrics())
    thresholds: Dict[str, float] = field(default_factory=dict)

    def resolve_pack(self) -> SafetyPack:
        """Resolve this task's SafetyPack, honoring pack_version if set."""
        if self.pack_version:
            return resolve_pack(self.pack_name, self.pack_version)
        return resolve_pack(self.pack_name)


class SafetyEvalRegistry:
    """In-memory registry for safety evaluation tasks."""

    _tasks: Dict[str, SafetyEvalTask] = {}

    # ------------------------------------------------------------------
    # Registration / lookup
    # ------------------------------------------------------------------
    @classmethod
    def register_task(cls, task: SafetyEvalTask) -> SafetyEvalTask:
        cls._tasks[task.name] = task
        return task

    @classmethod
    def register_default_tasks(cls) -> None:
        """Register a small set of default safety tasks if not already present."""
        if cls._tasks:
            # Assume explicit tasks already registered by user.
            return
        cls.register_task(
            SafetyEvalTask(
                name="harmbench_default",
                pack_name="harmbench",
                pack_version="1.0.0",
                judge_backend="local",
                thresholds={},
            )
        )
        cls.register_task(
            SafetyEvalTask(
                name="jailbreakbench_default",
                pack_name="jailbreakbench",
                pack_version="1.0.0",
                judge_backend="local",
                thresholds={},
            )
        )
        cls.register_task(
            SafetyEvalTask(
                name="xstest_default",
                pack_name="xstest",
                pack_version="1.0.0",
                judge_backend="local",
                thresholds={},
            )
        )
        cls.register_task(
            SafetyEvalTask(
                name="hh_rlhf_default",
                pack_name="hh_rlhf",
                pack_version="1.0.0",
                judge_backend="local",
                thresholds={},
            )
        )

    @classmethod
    def get_task(cls, name: str) -> SafetyEvalTask:
        if not cls._tasks:
            cls.register_default_tasks()
        if name not in cls._tasks:
            raise KeyError(f"Unknown safety eval task: {name}. Available: {list(cls._tasks.keys())}")
        return cls._tasks[name]

    @classmethod
    def list_tasks(cls) -> List[str]:
        if not cls._tasks:
            cls.register_default_tasks()
        return sorted(cls._tasks.keys())


def run_safety_eval_from_hf(
    task_name: str,
    *,
    max_samples: Optional[int] = None,
    split: Optional[str] = None,
    thresholds: Optional[Dict[str, float]] = None,
    harmfulness_scores: Optional[List[float]] = None,
    jailbreak_labels: Optional[List[bool]] = None,
) -> PackRunResult:
    """Load a Safety Pack from Hugging Face and run its pack runner.

    This function is intentionally simple: it delegates dataset loading to
    ``load_pack_from_hf`` and metric/gate computation to ``run_pack``.

    Args:
        task_name: Name of a registered SafetyEvalTask.
        max_samples: If set, truncate to this many rows.
        split: Optional override for HF split (defaults to pack mapping).
        thresholds: Optional gate thresholds to override those on the task.
        harmfulness_scores: Optional classifier scores for HarmBench-style packs.
        jailbreak_labels: Optional jailbreak labels for JailbreakBench-style packs.

    Returns:
        PackRunResult with metrics and gate decisions.
    """
    task = SafetyEvalRegistry.get_task(task_name)
    pack = task.resolve_pack()

    rows = load_pack_from_hf(pack.name, max_samples=max_samples, split=split)
    use_thresholds = thresholds if thresholds is not None else task.thresholds

    result = run_pack(
        pack.name,
        rows,
        pack_version=pack.version,
        thresholds=use_thresholds,
        harmfulness_scores=harmfulness_scores,
        jailbreak_labels=jailbreak_labels,
    )
    return result

