"""MultiJail and Aya Red-teaming: languages, row shape and dataset-table overrides. Offline."""

import pytest
from datasets import Dataset

import safetune
import safetune.config as sc
import safetune.data.dataset_ids as ids
from safetune.core.eval.pipeline import load_prompts
from safetune.data.loaders import load_aya_redteaming, load_multijail
from safetune.data.loaders.benchmarks import AYA_REDTEAMING_SPLITS, MULTIJAIL_LANGUAGES


def _jail(n, **cells):
    row = {"id": n, "source": "s", "tags": "['Violence', 'Theft']"}
    row.update({lang: f"p{n} {lang}" for lang in MULTIJAIL_LANGUAGES})
    row.update(cells)
    return row


def _aya(text):
    return {
        "prompt": text,
        "language": "Hindi",
        "harm_category": '["Bullying", "Graphic"]',
        "global_or_local": "local",
    }


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch):
    monkeypatch.setattr(sc, "_RUNTIME", sc.RuntimeConfig())


@pytest.fixture
def hub(monkeypatch):
    """``hub.rows`` is what every load returns, or a {split: rows} dict; calls are recorded."""

    class Hub:
        rows, calls = [], []

    Hub.calls = []

    def resolve(source, config_name=None, split=None, **kw):
        Hub.calls.append((source, split))

        class _Loader:
            def load(self):
                rows = Hub.rows[split] if isinstance(Hub.rows, dict) else Hub.rows
                return Dataset.from_list(rows)

        return _Loader()

    monkeypatch.setattr(ids.LoaderResolver, "resolve", staticmethod(resolve))
    return Hub


class TestMultiJail:
    def test_a_row_per_prompt_and_language(self, hub):
        hub.rows = [_jail(1), _jail(2)]
        ds = load_multijail(["en", "bn"])
        assert ds["prompt"] == ["p1 en", "p1 bn", "p2 en", "p2 bn"]
        assert ds["language"] == ["en", "bn", "en", "bn"]
        assert ds["id"] == ["1", "1", "2", "2"]

    def test_defaults_to_all_ten_languages(self, hub):
        hub.rows = [_jail(1)]
        assert load_multijail()["language"] == list(MULTIJAIL_LANGUAGES)

    def test_a_single_code_is_not_read_as_letters(self, hub):
        hub.rows = [_jail(1)]
        assert load_multijail("bn")["language"] == ["bn"]

    def test_categories_are_parsed_from_python_literal_text(self, hub):
        hub.rows = [_jail(1)]
        assert load_multijail("en")["category"] == ["Violence, Theft"]

    def test_a_blank_cell_is_skipped(self, hub):
        hub.rows = [_jail(1, bn="  "), _jail(2)]
        assert load_multijail(["bn"])["prompt"] == ["p2 bn"]

    def test_a_language_it_lacks_is_an_error(self, hub):
        with pytest.raises(ValueError, match=r"MultiJail has no \['hi'\]"):
            load_multijail(["en", "hi"])
        assert hub.calls == []


class TestAyaRedteaming:
    def test_each_language_loads_its_own_split(self, hub):
        hub.rows = {"hindi": [_aya("hi one")], "arabic": [_aya("ar one"), _aya("ar two")]}
        ds = load_aya_redteaming(["hi", "ar"])
        assert hub.calls == [
            ("CohereLabs/aya_redteaming", "hindi"),
            ("CohereLabs/aya_redteaming", "arabic"),
        ]
        assert ds["prompt"] == ["hi one", "ar one", "ar two"]
        assert ds["language"] == ["hi", "ar", "ar"]

    def test_defaults_to_all_eight_splits_the_dataset_card_names(self, hub):
        hub.rows = [_aya("x")]
        load_aya_redteaming()
        assert sorted(split for _, split in hub.calls) == [
            "arabic",
            "english",
            "filipino",
            "french",
            "hindi",
            "russian",
            "serbian",
            "spanish",
        ]

    def test_keeps_every_category(self, hub):
        hub.rows = [_aya("x")]
        row = load_aya_redteaming("hi")[0]
        assert row["category"] == "Bullying, Graphic"
        assert row["global_or_local"] == "local"

    def test_a_language_it_lacks_is_an_error(self, hub):
        with pytest.raises(ValueError, match=r"Aya Red-teaming has no \['bn'\]"):
            load_aya_redteaming("bn")

    def test_a_split_in_the_table_loads_only_that_split(self, hub):
        """A local single-file copy has one split, so the language loop must not run on it."""
        safetune.configure(
            datasets={"aya_redteaming": {"source": "aya_hi.jsonl", "split": "train"}}
        )
        hub.rows = [_aya("x")]
        ds = load_aya_redteaming()
        assert hub.calls == [("aya_hi.jsonl", "train")]
        assert ds["language"] == ["Hindi"]

    def test_the_split_argument_does_the_same(self, hub):
        hub.rows = {"hindi": [_aya("x")]}
        load_aya_redteaming(split="hindi")
        assert hub.calls == [("CohereLabs/aya_redteaming", "hindi")]


class TestLoadPrompts:
    def test_multijail_rows_link_a_prompts_translations(self, hub):
        hub.rows = [_jail(7)]
        en, sw = load_prompts("multijail", {"languages": ["en", "sw"]})
        assert (en["behavior_id"], sw["behavior_id"]) == ("7-en", "7-sw")
        assert en["multijail_id"] == sw["multijail_id"] == "7"
        assert (en["source"], en["language"], en["prompt"]) == ("multijail", "en", "p7 en")

    def test_aya_behavior_ids_are_unique_across_languages(self, hub):
        hub.rows = {"hindi": [_aya("a"), _aya("b")], "arabic": [_aya("c")]}
        rows = load_prompts("aya_redteaming", {"languages": ["hi", "ar"]})
        assert [r["behavior_id"] for r in rows] == ["hi-0", "hi-1", "ar-0"]
        assert rows[0]["aya_category"] == "Bullying, Graphic"

    @pytest.mark.parametrize("name", ["multijail", "aya_redteaming"])
    def test_max_prompts(self, hub, name):
        hub.rows = [_jail(1), _jail(2)] if name == "multijail" else [_aya("a"), _aya("b")]
        assert len(load_prompts(name, {"max_prompts": 3})) == 3


def test_both_are_registered_as_multilingual_jailbreak_benchmarks():
    from safetune.evaluate.suite.benchmarks import get_benchmark

    for name in ("multijail", "aya_redteaming"):
        spec = get_benchmark(name)
        assert spec.category == "jailbreak"
        assert "multilingual" in spec.tags
        assert "judge" in spec.description  # an English-only judge inflates the ASR


def test_the_splits_cover_the_languages_the_card_lists():
    assert set(AYA_REDTEAMING_SPLITS) == {"ar", "en", "es", "fr", "hi", "ru", "sr", "tl"}
