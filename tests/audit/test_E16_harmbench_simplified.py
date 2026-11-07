"""Audit E16: the HarmBench loader is Simplified (contextual rows without prompt_contextual, copyright rows by the classifier instead of the hash check)."""
from pathlib import Path

import safetune.data.loaders.benchmarks as LB
import safetune.core.eval.pipeline.loaders as PL

ROOT = Path(LB.__file__).resolve().parents[4]


def _flat(t):
    return " ".join(t.split())


def test_loader_docstrings_say_simplified_and_what_the_original_does():
    for doc in (_flat(LB.load_harmbench.__doc__), _flat(PL.load_harmbench.__doc__)):
        assert "Simplified" in doc
    d = _flat(LB.load_harmbench.__doc__)
    for needle in ("prompt_contextual", "eval_utils.py:401-403", "hash check", "eval_utils.py:367-391", "evaluate_completions.py:78"):
        assert needle in d, needle


def test_feature_map_and_scope_label_it():
    assert "| Loader: harmbench | Simplified |" in (ROOT / "docs/reference/feature-map.md").read_text()
    sc = _flat((ROOT / "docs/community/scope.md").read_text())
    assert "load_prompts('harmbench')" in sc and "prompt_contextual" in sc and "hash check" in sc


def test_the_loader_still_returns_the_same_rows(monkeypatch):
    """No code change: context is joined to the behavior as before."""
    from datasets import Dataset
    tables = {"standard": [{"prompt": "PLACEHOLDER s", "category": "c"}],
              "contextual": [{"prompt": "PLACEHOLDER q", "context": "PLACEHOLDER ctx", "category": "c"}],
              "copyright": [{"prompt": "PLACEHOLDER c", "category": "c"}]}
    monkeypatch.setattr(LB, "load", lambda name, config=None, split=None, **kw: Dataset.from_list(tables[config]))
    rows = list(LB.load_harmbench())
    assert [r["prompt"] for r in rows] == ["PLACEHOLDER s", "PLACEHOLDER ctx\n\n---\n\nPLACEHOLDER q", "PLACEHOLDER c"]
    assert [r["subset"] for r in rows] == ["standard", "contextual", "copyright"]
