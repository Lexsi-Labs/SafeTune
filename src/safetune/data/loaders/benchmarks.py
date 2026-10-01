"""Safety and capability benchmark dataset loaders (HF Hub).

All loaders normalize the output so the 'prompt' field always contains the
text to feed to the model under evaluation. This keeps _compute_metrics
backend-agnostic.

This is the one loader per benchmark: ``evaluate()``, ``trainer.evaluate()`` and
``load_prompts()`` all go through it. Loaders that take ``**overrides`` pass those
dataset-table fields (``source``, ``config``, ``split``, ``where``, ...) to
:func:`safetune.data.dataset_ids.load` as per-call overrides.
"""
from __future__ import annotations

from safetune.data.dataset_ids import load, spec


# ── Safety benchmarks ──────────────────────────────────────────────────────────
# Sources, configs and splits come from safetune.data.dataset_ids (overridable via
# safetune.configure(datasets=...)). Arguments given here override the table.

def load_harmbench(split: str = None, subset=None, **overrides):
    """HarmBench (Mazeika et al., 2024): the 400 text behaviours.

    Default: walledai/HarmBench, configs 'standard' (200), 'contextual' (100)
    and 'copyright' (100), in that order, split 'train'. A contextual row's
    context goes before its behaviour, as HarmBench's DirectRequest baseline
    builds it: ``f"{context}\n\n---\n\n{behavior}"``. ``subset``: one config
    or a list; ``subset="standard"`` (or ``configure(datasets={"harmbench":
    {"config": "standard"}})``) is the 200-behaviour set used before.
    Returns a Dataset with 'prompt', 'category' and 'subset' (the config).
    """
    from datasets import Dataset
    configs = subset or overrides.pop("config", None) or spec("harmbench").get("config") or [None]
    rows = []
    for config in [configs] if isinstance(configs, str) else configs:
        for ex in load("harmbench", split=split, **{**overrides, "config": config}):
            prompt = ex.get("prompt", ex.get("Behavior"))
            if ex.get("context"):
                prompt = f"{ex['context']}\n\n---\n\n{prompt}"
            rows.append({"prompt": prompt, "category": ex.get("category") or "",
                         "subset": config or ""})
    return Dataset.from_list(rows)


def load_wildjailbreak(split: str = None, config: str = None, **overrides):
    """WildJailbreak eval set (allenai/wildjailbreak).

    The 'eval' config 'train' split contains the adversarial prompts.
    'adversarial_prompt' is renamed to 'prompt'.
    """
    ds = load("wildjailbreak", split=split, config=config, **overrides)
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


def load_sorrybench(split: str = None, **overrides):
    """SorryBench v1 (sorry-bench/sorry-bench-202503).

    Keeps rows matching the spec's ``where`` filter (default prompt_style='base';
    set e.g. ``{"where": {"prompt_style": "translate-fr"}}`` for translated
    prompts) and flattens the 'turns' list to a flat 'prompt' string.
    """
    ds = load("sorrybench_v1", split=split, **overrides)
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

    Default: the Hammington/hexphi mirror (200 prompts, stable schema, a
    'prompt' column). The primary LLM-Tuning-Safety/HEx-PHI has per-category
    CSVs with inconsistent column names.
    """
    return load("hexphi")


def load_ailuminate(csv_url: str = None):
    """MLCommons AILuminate DEMO — 1200 prompts, 12 hazard categories.

    Default: the official GitHub CSV. Returns a list of dicts with 'prompt',
    'hazard' and 'persona' fields.
    """
    ds = load("ailuminate", source=csv_url)
    return [{"prompt": item.get("prompt") or "",
             "hazard": item.get("hazard") or "",
             "persona": item.get("persona") or ""} for item in ds]


def load_advbench(split: str = None, **overrides):
    """AdvBench harmful behaviors (Zou et al., 2023)."""
    return load("advbench", split=split, **overrides)


def load_orbench(splits: list = None, **overrides):
    """OR-Bench benchmark (bench-llm/or-bench) — both splits combined.

    Returns a list of dicts with 'prompt' and 'split' fields so pipeline.py
    can separate hard-1k (over-refusal metric) from toxic (safety metric).

    splits (HF configs; default from the table):
      or-bench-hard-1k (1319 prompts): safe-but-superficially-risky → model SHOULD comply
      or-bench-toxic   (655 prompts):  genuinely toxic → model SHOULD refuse
    """
    configs = splits or spec("orbench").get("config") or [None]
    if isinstance(configs, str):
        configs = [configs]
    rows = []
    for split_name in configs:
        for item in load("orbench", **{**overrides, "config": split_name}):
            prompt = item.get("prompt") or item.get("question") or item.get("instruction", "")
            rows.append({"prompt": prompt, "split": split_name or item.get("split", ""),
                         "category": item.get("category", "")})
    return rows


def load_orbench_hard(**overrides):
    """OR-Bench hard-1k (1,319 benign prompts that sound harmful): over-refusal,
    so a lower refusal rate is better."""
    return load_orbench(["or-bench-hard-1k"], **overrides)


def load_orbench_toxic(**overrides):
    """OR-Bench toxic (655 harmful prompts that should be refused)."""
    return load_orbench(["or-bench-toxic"], **overrides)


def load_jailbreakbench(split: str = None, **overrides):
    """JailbreakBench (Chao et al., 2024): JBB-Behaviors, the 100 harmful
    behaviors ('Goal' is renamed to 'prompt')."""
    return load("jailbreakbench", split=split, **overrides)


def load_xstest(split: str = None, **overrides):
    """XSTest over-refusal benchmark (Röttger et al., 2023)."""
    return load("xstest", split=split, **overrides)


def load_beavertails(split: str = None):
    """BeaverTails (PKU-Alignment)."""
    return load("beavertails", split=split)


def load_star1(split: str = None):
    """STAR-1 safety training data (UCSC-VLAA/STAR-1); 'question' is renamed to 'prompt'."""
    return load("star1", split=split)


# ── Unlearning and multi-turn benchmarks ───────────────────────────────────────

def load_muse(domain: str = "news", split: str = None, config: str = None, **overrides):
    """MUSE (Shi et al., arXiv:2407.06460): MUSE-News (``domain="news"``) or
    MUSE-Books. Default: the 'verbmem' forget set, 100 passages in 'prompt' with
    their true continuation in 'gt'."""
    if domain not in ("news", "books"):
        raise ValueError(f"MUSE domain must be 'news' or 'books', got {domain!r}")
    return load(f"muse_{domain}", split=split, config=config, **overrides)


def load_rwku(split: str = None, config: str = None, **overrides):
    """RWKU (Jin et al., arXiv:2406.10890). Default: the 'forget_level2'
    question-answer probes about the 200 forget targets ('query' is renamed to
    'prompt'; 'answer' and 'subject' are kept)."""
    return load("rwku", split=split, config=config, **overrides)


def load_safedialbench(split: str = None, **overrides):
    """SafeDialBench (Zhao et al., arXiv:2502.11090): multi-turn safety dialogues,
    from ``HongyeCao/SafeDialBench`` (the paper's own dataset release; the
    ``thu-coai`` id from the paper's Hub link 404s).

    Each row gets 'conversation' (a list of {role, content}) and 'prompt': the
    turns up to the last user turn flattened into one message, since
    ``evaluate()`` generates single-turn. That is an approximation of the
    multi-turn protocol. The real Hub schema stores each exchange as a
    {"user": ..., "bot": ...} pair under 'history' (an incomplete last pair,
    with no 'bot' yet, is the attack turn itself); a 'conversation' /
    'dialogue' / 'messages' key (as {role, content} turns, or plain strings)
    is honored first, for a copy pointed at with
    ``configure(datasets={"safedialbench": ...})``.
    """
    rows = []
    for item in load("safedialbench", split=split, **overrides):
        raw = item.get("conversation") or item.get("dialogue") or item.get("messages")
        if raw is None and item.get("history"):
            raw = []
            for turn in item["history"]:
                if isinstance(turn, dict) and ("user" in turn or "bot" in turn):
                    if turn.get("user"):
                        raw.append({"role": "user", "content": str(turn["user"])})
                    if turn.get("bot"):
                        raw.append({"role": "assistant", "content": str(turn["bot"])})
                else:
                    raw.append(turn)
        raw = raw or []
        if isinstance(raw, str):
            raw = [raw]
        conversation = [
            {"role": str(t.get("role", "user")), "content": str(t.get("content", t.get("text", "")))}
            if isinstance(t, dict) else {"role": ("user", "assistant")[i % 2], "content": str(t)}
            for i, t in enumerate(raw)
        ]
        last_user = max((i for i, t in enumerate(conversation) if t["role"] == "user"), default=-1)
        prompt = "\n\n".join(f"{t['role']}: {t['content']}" for t in conversation[:last_user + 1])
        rows.append({**item, "conversation": conversation, "prompt": item.get("prompt") or prompt})
    return rows


# ── Capability / utility benchmarks ───────────────────────────────────────────

def load_mmlu(subset: str = None, split: str = None):
    """MMLU capability retention (Hendrycks et al., 2021)."""
    return load("mmlu", split=split, config=subset)


def load_gsm8k(split: str = None, config: str = None):
    """GSM8K math word problems (Cobbe et al., 2021).

    'question' column renamed to 'prompt' for uniform access.
    """
    return load("gsm8k", split=split, config=config)


def load_humaneval(split: str = None):
    """HumanEval coding benchmark (Chen et al., 2021).

    'prompt' column already contains the docstring + function signature.
    """
    return load("humaneval", split=split)


def load_medmcqa(split: str = None):
    """MedMCQA medical multiple-choice benchmark (Pal et al., 2022).

    'question' column renamed to 'prompt'.
    """
    return load("medmcqa", split=split)


__all__ = [
    # Safety
    "load_harmbench",
    "load_wildjailbreak",
    "load_sorrybench",
    "load_hexphi",
    "load_ailuminate",
    "load_advbench",
    "load_orbench",
    "load_orbench_hard",
    "load_orbench_toxic",
    "load_jailbreakbench",
    "load_xstest",
    "load_beavertails",
    "load_star1",
    "load_muse",
    "load_rwku",
    "load_safedialbench",
    # Capability
    "load_mmlu",
    "load_gsm8k",
    "load_humaneval",
    "load_medmcqa",
]
