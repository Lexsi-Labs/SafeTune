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
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..data_compiler.safety_packs import SafetyPack, resolve_pack
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
