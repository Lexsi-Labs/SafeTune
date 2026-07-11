"""Safety evaluation harness (re-exports + unified evaluate entry point)."""

from .evaluate import evaluate
from .tamperbench import TamperBenchEvaluator
from .judge import JudgeAdapter, LocalClassifierJudge, APIJudge
from .benchmarks import (
    BenchmarkSpec,
    CATEGORIES,
    REGISTRY,
    get_benchmark,
    list_benchmarks,
    benchmarks_by_category,
)
from .spectral import SpectralEntropyMonitor, SpectralMonitorConfig

__all__ = [
    "evaluate",
    "TamperBenchEvaluator",
    "JudgeAdapter",
    "LocalClassifierJudge",
    "APIJudge",
    "BenchmarkSpec",
    "CATEGORIES",
    "REGISTRY",
    "get_benchmark",
    "list_benchmarks",
    "benchmarks_by_category",
    "SpectralEntropyMonitor",
    "SpectralMonitorConfig",
]
