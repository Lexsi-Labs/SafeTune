"""SafeTune's dataset table: every built-in dataset by short name.

Each entry is a spec dict:

* ``source``: HF repo id, local ``.jsonl``/``.json``/``.csv``/``.parquet`` file or
  directory, or an http(s) URL to a CSV/JSON file.
* ``config``: HF config name (HarmBench, OR-Bench: a list, one load per config).
* ``split``: split to load. Local files use it when present, else ``train``.
* ``prompt_col`` / ``response_col``: renamed to ``prompt`` / ``response`` at load.
* ``where``: equality filter, e.g. ``{"prompt_style": "base"}``.
* ``limit``: keep only the first N rows.

Override from Python or the CLI YAML ``datasets:`` block::

    safetune.configure(datasets={
        "harmbench": "harmbench_hi.jsonl",                          # str = new source
        "sorrybench_v1": {"where": {"prompt_style": "translate-fr"}},  # dict without source: merged over the default
        "beavertails": {"source": "myorg/bt-hi", "split": "train"},   # dict with source: replaces the default
    })

Overrides must have the columns the call site reads: a prompt column for eval
benchmarks (``prompt``, or name it with ``prompt_col``); ``prompt``/``response``/
``is_safe`` for ``beavertails``; ``instruction``/``input``/``output`` for
``alpaca``; ``prompt``/``response`` for ``safety_refusals``; the columns in
``runner.utils.data_utils._DOMAIN_COLUMNS`` for ``sft_*``. Missing columns raise
``KeyError``.

Specs are read when a loader runs, so ``configure()`` after import works. The
repo-id constants below stay for backward compatibility.
"""
from __future__ import annotations

from safetune.data.loaders.resolver import LoaderResolver

BEAVERTAILS = "PKU-Alignment/BeaverTails"
ALPACA = "tatsu-lab/alpaca"
GSM8K = "openai/gsm8k"
COMPETITION_MATH = "hendrycks/competition_math"
CODE_ALPACA = "sahil2801/CodeAlpaca-20k"
DOLLY = "databricks/databricks-dolly-15k"
CHATDOCTOR = "lavita/ChatDoctor-HealthCareMagic-100k"
LEGAL_QA = "dzunggg/legal-qa-v1"
ADVBENCH = "walledai/AdvBench"
HARMBENCH = "walledai/HarmBench"
AILUMINATE_URL = ("https://raw.githubusercontent.com/mlcommons/ailuminate/main/"
                  "airr_official_1.0_demo_en_us_prompt_set_release.csv")

DATASETS: dict[str, dict] = {
    # Safety benchmarks: evaluate(), trainer.evaluate(), load_bench_prompts()
    # The SafeTune paper's suite sizes: HarmBench 400 = standard 200 + contextual
    # 100 + copyright 100 (a config list: one load per config, in this order);
    # WildJailbreak 500 = the first 500 adversarial_harmful rows of the eval set
    # (load_prompts()' documented selection); OR-Bench hard-1k 1,319 (the paper says 1,320) and toxic 655,
    # reported as separate benchmarks (orbench_hard / orbench_toxic in evaluate()).
    "harmbench":      {"source": HARMBENCH, "config": ["standard", "contextual", "copyright"],
                       "split": "train"},
    "advbench":       {"source": ADVBENCH, "split": "train"},
    "wildjailbreak":  {"source": "allenai/wildjailbreak", "config": "eval", "split": "train",
                       "where": {"data_type": "adversarial_harmful"}, "limit": 500},
    "sorrybench_v1":  {"source": "sorry-bench/sorry-bench-202503", "split": "train",
                       "where": {"prompt_style": "base"}},
    "hexphi":         {"source": "Hammington/hexphi", "split": "train"},
    "ailuminate":     {"source": AILUMINATE_URL, "prompt_col": "prompt_text"},
    "orbench":        {"source": "bench-llm/or-bench", "split": "train",
                       "config": ["or-bench-hard-1k", "or-bench-toxic"]},
    "jailbreakbench": {"source": "JailbreakBench/JBB-Behaviors", "config": "behaviors",
                       "split": "harmful", "prompt_col": "Goal"},
    "xstest":         {"source": "walledai/XSTest", "split": "test"},
    "star1":          {"source": "UCSC-VLAA/STAR-1", "split": "train", "prompt_col": "question"},
    "hh-rlhf":        {"source": "Anthropic/hh-rlhf", "split": "test"},
    # Capability benchmarks
    "mmlu":           {"source": "cais/mmlu", "config": "all", "split": "test",
                       "prompt_col": "question"},
    "gsm8k":          {"source": GSM8K, "config": "main", "split": "test", "prompt_col": "question"},
    "humaneval":      {"source": "openai_humaneval", "split": "test"},
    "medmcqa":        {"source": "openlifescienceai/medmcqa", "split": "validation",
                       "prompt_col": "question"},
    "competition_math": {"source": COMPETITION_MATH, "split": "train"},  # BOLT baseline
    "mbpp":           {"source": "mbpp", "split": "train"},              # BOLT baseline
    # Calibration, contamination and training sets
    "beavertails":    {"source": BEAVERTAILS, "split": "30k_train"},
    "alpaca":         {"source": ALPACA, "split": "train"},
    "safety_refusals": {"source": None},  # None = the built-in 6 English refusal pairs
    "sft_gsm8k":      {"source": GSM8K, "config": "main", "split": "train"},
    "sft_code":       {"source": CODE_ALPACA, "split": "train"},
    "sft_dolly":      {"source": DOLLY, "split": "train"},
    "sft_medical":    {"source": CHATDOCTOR, "split": "train"},
    "sft_legal":      {"source": LEGAL_QA, "split": "train"},
    # Unlearning / multi-turn benchmarks (evaluate() registry) and the extra
    # load_prompts() benchmarks in safetune.core.eval.pipeline
    "strongreject":   {"source": "walledai/StrongREJECT", "split": "train"},
    "agentharm":      {"source": "ai-safety-institute/AgentHarm", "config": "harmful",
                       "split": "test_public"},
    "cares":          {"source": "HFXM/CARES-18K", "split": "test"},
    "airbench":       {"source": "stanford-crfm/air-bench-2024", "config": "default",
                       "split": "test"},
    "saladbench":     {"source": "OpenSafetyLab/Salad-Data", "config": "base_set",
                       "split": "train"},
    "muse_news":      {"source": "muse-bench/MUSE-News", "config": "verbmem", "split": "forget"},
    "muse_books":     {"source": "muse-bench/MUSE-Books", "config": "verbmem", "split": "forget"},
    "rwku":           {"source": "jinzhuoran/RWKU", "config": "forget_level2", "split": "test",
                       "prompt_col": "query"},  # QA probes; forget_target has only target names
    # thu-coai/SafeDialBench (the id in the paper's own Hub link) 404s; this is
    # the paper's dataset release under the first author's own account, live
    # and public, split "train" (it has no other split).
    "safedialbench":  {"source": "HongyeCao/SafeDialBench", "split": "train"},
}


def _overrides() -> dict:
    from safetune.config import get_config
    return get_config().datasets


def spec(name: str) -> dict:
    """Effective spec for ``name``: the ``configure(datasets=...)`` override, if any,
    else the built-in entry."""
    override = _overrides().get(name)
    if override is None:
        if name not in DATASETS:
            from safetune.config import _did_you_mean
            raise KeyError(f"unknown dataset {name!r}{_did_you_mean(name, DATASETS)}")
        return dict(DATASETS[name])
    if isinstance(override, str):
        return {"source": override}
    if "source" in override or name not in DATASETS:
        return dict(override)
    return {**DATASETS[name], **override}


def effective_specs() -> dict:
    """Every dataset name → its effective spec (recorded in results JSON)."""
    return {name: spec(name) for name in {**DATASETS, **_overrides()}}


def _pick_split(ds, name: str, split):
    """A local/multi-file load can come back as a {split: Dataset} dict;
    resolve it to one Dataset the same way regardless of caller."""
    from datasets import DatasetDict
    if not isinstance(ds, DatasetDict):
        return ds
    pick = split if split in ds else "train" if "train" in ds else (
        next(iter(ds)) if len(ds) == 1 else None)
    if pick is None:  # e.g. a folder of several files: never pick one silently
        raise ValueError(f"dataset {name!r}: split {split!r} not found; it has "
                         f"{sorted(ds)}. Pass the split (or the file) to load.")
    return ds[pick]


def load(name: str, **per_call):
    """Load dataset ``name`` from the table, or treat ``name`` as a source
    (HF id / local file / URL / dataset folder, default split ``train``) if it is
    not in the table. A folder with a ``README.md`` (an HF dataset folder, e.g. a
    CuratorKIT export) loads as ``load_dataset(folder, config, split=split)``.
    A ``config`` that is a list of names loads and concatenates each one, in
    order (e.g. HarmBench's ``standard`` + ``contextual`` + ``copyright``,
    or OR-Bench's ``or-bench-hard-1k`` + ``or-bench-toxic``).

    ``per_call`` spec fields (``source``, ``config``, ``split``, ``where``, ...)
    override the table for this call; ``None`` values are ignored. Returns a
    ``datasets.Dataset``.
    """
    from datasets import concatenate_datasets, load_dataset
    from safetune.utils.errors import hf_access_errors

    known = name in DATASETS or name in _overrides()
    s = spec(name) if known else {"source": name, "split": "train"}
    s.update({k: v for k, v in per_call.items() if v is not None})
    source, split, config = s["source"], s.get("split"), s.get("config")
    if source.startswith(("http://", "https://")):
        fmt = "json" if source.endswith((".json", ".jsonl")) else "csv"
        ds = load_dataset(fmt, data_files=source, split="train")
    elif isinstance(config, (list, tuple)):
        with hf_access_errors(source, kind="dataset"):
            parts = [_pick_split(LoaderResolver.resolve(source, config_name=c, split=split).load(),
                                 name, split) for c in config]
        ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    else:
        with hf_access_errors(source, kind="dataset"):
            ds = _pick_split(LoaderResolver.resolve(source, config_name=config, split=split).load(),
                             name, split)
    for col, std in ((s.get("prompt_col"), "prompt"), (s.get("response_col"), "response")):
        if col and col != std and std not in ds.column_names:
            if col not in ds.column_names:
                raise KeyError(f"dataset {name!r}: no column {col!r}; have {ds.column_names}")
            ds = ds.rename_column(col, std)
    where = s.get("where")
    if where:
        ds = ds.filter(lambda ex: all(ex[k] == v for k, v in where.items()))
    if s.get("limit") is not None:
        ds = ds.select(range(min(int(s["limit"]), len(ds))))
    return ds
