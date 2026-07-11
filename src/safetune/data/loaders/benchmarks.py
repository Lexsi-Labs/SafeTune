"""Safety and capability benchmark dataset loaders (HF Hub).

All loaders normalize the output so the 'prompt' field always contains the
text to feed to the model under evaluation. This keeps _compute_metrics
backend-agnostic.
"""
from __future__ import annotations


def _require_datasets():
    try:
        import datasets
        return datasets
    except ImportError as e:
        raise ImportError("Install `datasets`: pip install datasets") from e


# ── Safety benchmarks ──────────────────────────────────────────────────────────

def load_harmbench(split: str = "train", subset: str = "standard"):
    """HarmBench (Mazeika et al., 2024).

    walledai/HarmBench 'train' split contains the standard 400 behaviors.
    'Behavior' column is renamed to 'prompt' for uniform downstream access.
    """
    ds = _require_datasets().load_dataset("walledai/HarmBench", subset, split=split)
    if "Behavior" in ds.column_names:
        ds = ds.rename_column("Behavior", "prompt")
    return ds


def load_wildjailbreak(split: str = "train", config: str = "eval"):
    """WildJailbreak eval set (allenai/wildjailbreak).

    The 'eval' config 'train' split contains the 500 adversarial prompts.
    'adversarial_prompt' is renamed to 'prompt'.
    """
    ds = _require_datasets().load_dataset("allenai/wildjailbreak", config, split=split)
    # The eval config exposes the prompt under different column names across
    # dataset revisions: 'adversarial_prompt', 'adversarial', or 'vanilla_prompt'.
    if "prompt" not in ds.column_names:
        for col in ("adversarial_prompt", "adversarial", "vanilla_prompt"):
            if col in ds.column_names:
                ds = ds.rename_column(col, "prompt")
                break
    if "prompt" not in ds.column_names:
        raise ValueError(
            f"load_wildjailbreak: no prompt column found; have {ds.column_names}")
    return ds


def load_sorrybench(split: str = "train"):
    """SorryBench v1 (sorry-bench/sorry-bench-202503).

    Filters to prompt_style='base' (single-turn) and flattens the 'turns'
    list to a flat 'prompt' string.
    """
    datasets = _require_datasets()
    ds = datasets.load_dataset("sorry-bench/sorry-bench-202503", split=split)
    # Filter to single-turn base prompts only
    if "prompt_style" in ds.column_names:
        ds = ds.filter(lambda x: x.get("prompt_style") == "base")
    # Flatten turns[0]['content'] → 'prompt'
    if "turns" in ds.column_names and "prompt" not in ds.column_names:
        def _extract(ex):
            turns = ex.get("turns") or []
            text = turns[0]["content"] if turns and isinstance(turns[0], dict) else str(turns[0]) if turns else ""
            return {"prompt": text}
        ds = ds.map(_extract)
    return ds


def load_hexphi():
    """HEx-PHI — harmful instructions benchmark (Qi et al., 2023).

    Uses Hammington/hexphi mirror (200 prompts, stable schema).
    The primary LLM-Tuning-Safety/HEx-PHI has per-category CSVs with
    inconsistent column names, making it unreliable to load as a single dataset.
    """
    ds = _require_datasets().load_dataset("Hammington/hexphi", split="train")
    # Mirror already has a 'prompt' column.
    return ds


def load_ailuminate(
    csv_url: str = "https://raw.githubusercontent.com/mlcommons/ailuminate/main/airr_official_1.0_demo_en_us_prompt_set_release.csv",
):
    """MLCommons AILuminate DEMO — 1290 prompts, 12 hazard categories.

    Loads from the official GitHub CSV. Returns a list of dicts with 'prompt' field.
    """
    import csv
    import io
    import urllib.request
    with urllib.request.urlopen(csv_url) as resp:
        text = resp.read().decode("utf-8")
    rows = []
    for item in csv.DictReader(io.StringIO(text)):
        rows.append({
            "prompt": item.get("prompt_text", ""),
            "hazard": item.get("hazard", ""),
            "persona": item.get("persona", ""),
        })
    return rows


def load_advbench(split: str = "train"):
    """AdvBench harmful behaviors (Zou et al., 2023)."""
    return _require_datasets().load_dataset("walledai/AdvBench", split=split)


def load_orbench(splits: list = None):
    """OR-Bench benchmark (bench-llm/or-bench) — both splits combined.

    Returns a list of dicts with 'prompt' and 'split' fields so pipeline.py
    can separate hard-1k (over-refusal metric) from toxic (safety metric).

    splits:
      or-bench-hard-1k (1320 prompts): safe-but-superficially-risky → model SHOULD comply
      or-bench-toxic   (655 prompts):  genuinely toxic → model SHOULD refuse
    """
    datasets_lib = _require_datasets()
    if splits is None:
        splits = ["or-bench-hard-1k", "or-bench-toxic"]
    rows = []
    for split_name in splits:
        ds = datasets_lib.load_dataset("bench-llm/or-bench", split_name, split="train")
        for item in ds:
            prompt = item.get("prompt") or item.get("question") or item.get("instruction", "")
            rows.append({"prompt": prompt, "split": split_name,
                         "category": item.get("category", "")})
    return rows


def load_jailbreakbench(split: str = "harmful"):
    """JailbreakBench (Chao et al., 2024)."""
    return _require_datasets().load_dataset(
        "JailbreakBench/JBB-Behaviors", "behaviors", split=split
    )


def load_xstest(split: str = "test"):
    """XSTest over-refusal benchmark (Röttger et al., 2023)."""
    return _require_datasets().load_dataset("walledai/XSTest", split=split)


def load_beavertails(split: str = "30k_train"):
    """BeaverTails (PKU-Alignment)."""
    return _require_datasets().load_dataset(
        "PKU-Alignment/BeaverTails", split=split
    )


def load_star1(split: str = "train"):
    """STAR-1 safety training data (UCSC-VLAA/STAR-1)."""
    return _require_datasets().load_dataset("UCSC-VLAA/STAR-1", split=split)


# ── Capability / utility benchmarks ───────────────────────────────────────────

def load_mmlu(subset: str = "all", split: str = "test"):
    """MMLU capability retention (Hendrycks et al., 2021)."""
    return _require_datasets().load_dataset("cais/mmlu", subset, split=split)


def load_gsm8k(split: str = "test", config: str = "main"):
    """GSM8K math word problems (Cobbe et al., 2021).

    'question' column renamed to 'prompt' for uniform access.
    """
    ds = _require_datasets().load_dataset("openai/gsm8k", config, split=split)
    if "question" in ds.column_names and "prompt" not in ds.column_names:
        ds = ds.rename_column("question", "prompt")
    return ds


def load_humaneval(split: str = "test"):
    """HumanEval coding benchmark (Chen et al., 2021).

    'prompt' column already contains the docstring + function signature.
    """
    return _require_datasets().load_dataset("openai_humaneval", split=split)


def load_medmcqa(split: str = "validation"):
    """MedMCQA medical multiple-choice benchmark (Pal et al., 2022).

    'question' column renamed to 'prompt'.
    """
    ds = _require_datasets().load_dataset("openlifescienceai/medmcqa", split=split)
    if "question" in ds.column_names and "prompt" not in ds.column_names:
        ds = ds.rename_column("question", "prompt")
    return ds


__all__ = [
    # Safety
    "load_harmbench",
    "load_wildjailbreak",
    "load_sorrybench",
    "load_hexphi",
    "load_ailuminate",
    "load_advbench",
    "load_orbench",
    "load_jailbreakbench",
    "load_xstest",
    "load_beavertails",
    "load_star1",
    # Capability
    "load_mmlu",
    "load_gsm8k",
    "load_humaneval",
    "load_medmcqa",
]
