"""Audit E20: CARES-18K rows carry a polarity (harmful_level 0 = harmless). Tiny fake table shaped like the dataset card; no network."""
import pytest
from datasets import Dataset

from safetune.core.eval.pipeline import loaders as L
from safetune.core.eval.pipeline.loaders import load_prompts


def _rows(levels):
    return [{"principle_index": 1, "generation_model": "m", "harmful_level": lv, "method": "direct",
             "base_prompt": "PLACEHOLDER base %d" % i, "prompt": "PLACEHOLDER prompt %d" % i} for i, lv in enumerate(levels)]


def test_polarity_follows_harmful_level_and_no_row_is_dropped(monkeypatch):
    monkeypatch.setattr(L, "_load", lambda name, cfg: Dataset.from_list(_rows([0, 1, 2, 3, 0])))
    out = load_prompts("cares")
    assert [r["polarity"] for r in out] == ["harmless", "harmful", "harmful", "harmful", "harmless"]
    assert [r["cares_harmful_level"] for r in out] == [0, 1, 2, 3, 0] and len(out) == 5


@pytest.mark.parametrize("level", [None, "", "x", 4, -1])
def test_a_missing_or_out_of_range_level_is_unknown(monkeypatch, level):
    rows = _rows([0]); rows[0]["harmful_level"] = level
    monkeypatch.setattr(L, "_load", lambda name, cfg: Dataset.from_list(rows))
    assert load_prompts("cares")[0]["polarity"] == "unknown"


def test_existing_fields_are_unchanged(monkeypatch):
    monkeypatch.setattr(L, "_load", lambda name, cfg: Dataset.from_list(_rows([2, 0])))
    r = load_prompts("cares", {"max_prompts": 1})
    assert len(r) == 1 and r[0]["source"] == "cares" and r[0]["behavior_id"] == "0" and r[0]["prompt"] == "PLACEHOLDER prompt 0"
    assert r[0]["cares_base_prompt"] == "PLACEHOLDER base 0" and r[0]["cares_method"] == "direct" and r[0]["cares_principle_index"] == 1


def test_the_docstring_says_to_exclude_harmless_rows_from_asr():
    assert "exclude" in " ".join(L.load_cares.__doc__.split()).lower() and "polarity" in L.load_cares.__doc__
