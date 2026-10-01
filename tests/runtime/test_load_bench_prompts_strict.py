"""load_bench_prompts must not silently drop a benchmark under the strict default."""
import pytest

import safetune
from safetune.runner.utils import data_utils


def _gated(name):
    raise ConnectionError("Dataset 'walledai/HarmBench' is a gated dataset on the Hub.")


def test_strict_raises_with_gated_hint(monkeypatch):
    import safetune.evaluate.suite.benchmarks as b
    monkeypatch.setattr(b, "load_benchmark", _gated)
    with pytest.raises(RuntimeError, match="gated Hugging Face dataset.*configure\\(datasets="):
        data_utils.load_bench_prompts(["harmbench"])


def test_lenient_skips(monkeypatch):
    import safetune.evaluate.suite.benchmarks as b
    monkeypatch.setattr(b, "load_benchmark", _gated)
    safetune.configure(eval_strict=False)
    try:
        assert data_utils.load_bench_prompts(["harmbench"]) == {}
    finally:
        safetune.configure(eval_strict=True)
