"""Audit E12-E25: benchmark loaders. Fake `load` returning tiny tables shaped like the datasets' public metadata; no network.
Real-data fact checks at the bottom skip when the Hub is unreachable."""
import importlib

import pytest
from datasets import Dataset

from safetune.core.eval.pipeline.loaders import load_prompts

LB = importlib.import_module("safetune.data.loaders.benchmarks")
SB = importlib.import_module("safetune.instrumentation.evaluate.suite.benchmarks")
E = importlib.import_module("safetune.instrumentation.evaluate.suite.evaluate")


def _fake(monkeypatch, table):
    """table: {dataset name: {config or None: rows}}"""
    def load(name, source=None, config=None, split=None, **kw):
        rows = table[name]
        return Dataset.from_list(rows[config] if isinstance(rows, dict) else rows)
    monkeypatch.setattr(LB, "load", load)


def test_airbench_rows_have_a_unique_row_id_and_behavior_id_is_unchanged(monkeypatch):
    """Real data: 5,694 rows but 314 distinct `cate-idx` (up to 30 prompts share one); behavior_id keeps that value, row_id is unique."""
    from safetune.core.eval.pipeline import loaders as L
    rows = [{"cate-idx": "1.1.1", "l2-name": "a", "l3-name": "b", "l4-name": "c", "prompt": "PLACEHOLDER %d" % i} for i in range(3)]
    monkeypatch.setattr(L, "_load", lambda name, cfg: Dataset.from_list(rows))
    out = load_prompts("airbench")
    assert [r["behavior_id"] for r in out] == ["1.1.1"] * 3
    assert [r["row_id"] for r in out] == ["0", "1", "2"]
    assert [r["prompt"] for r in out] == [r["prompt"] for r in rows] and out[0]["airbench_category"] == "b"
    assert [r["row_id"] for r in load_prompts("airbench", {"max_prompts": 2})] == ["0", "1"]


@pytest.mark.parametrize("config,col", [("privleak", "text"), ("knowmem", "question")])
def test_muse_configs_without_a_prompt_column_raise(monkeypatch, config, col):
    _fake(monkeypatch, {"muse_news": [{col: "PLACEHOLDER passage %d" % i} for i in range(3)]})
    with pytest.raises(ValueError, match="no 'prompt' column"):
        load_prompts("muse", {"config": config, "split": "retain"})


def test_muse_default_verbmem_rows_are_unchanged(monkeypatch):
    """Already-correct path: verbmem forget rows (`prompt`, `gt`)."""
    _fake(monkeypatch, {"muse_news": [{"prompt": "PLACEHOLDER p%d" % i, "gt": "g"} for i in range(3)]})
    rows = load_prompts("muse", {})
    assert [r["prompt"] for r in rows] == ["PLACEHOLDER p0", "PLACEHOLDER p1", "PLACEHOLDER p2"]
    assert rows[0]["split"] == "forget" and rows[0]["domain"] == "news" and rows[0]["text"] == rows[0]["prompt"]


# ---------------------------------------------------------------- real-data facts (skip offline)

def _real(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:   # no Hub, no cache, gated
        pytest.skip(f"{type(e).__name__}: {str(e)[:80]}")


@pytest.mark.parametrize("name,n", [("jailbreakbench", 100), ("agentharm", 176), ("rwku", 2879)])
def test_row_counts_match_the_dataset_metadata(name, n):
    assert len(_real(load_prompts, name)) == n


def test_orbench_splits_match_the_dataset_card():
    rows = _real(load_prompts, "orbench")
    from collections import Counter
    assert Counter(r["orbench_split"] for r in rows) == {"or-bench-hard-1k": 1319, "or-bench-toxic": 655}
    assert len({r["behavior_id"] for r in rows}) == len(rows)
