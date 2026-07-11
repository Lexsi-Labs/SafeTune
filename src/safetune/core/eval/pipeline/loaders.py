"""
Benchmark prompt loaders.

Each loader returns a list of dicts with at minimum:

    {
        "source": "<benchmark-name>",
        "behavior_id": "<stable id within the benchmark>",
        "prompt": "<the prompt string>",
        ...benchmark-specific metadata...
    }

We keep the loader API uniform across benchmarks so the same Generator can
dispatch over any of them.

Default benchmark IDs follow the cthetha-eval choices (proven set):

* HarmBench         walledai/HarmBench, standard split, 200 prompts
* WildJailbreak     allenai/wildjailbreak, eval config, 500 adversarial
* SorryBench v1     sorry-bench/sorry-bench-202503, filter_prompt_style=base
* AdvBench          walledai/AdvBench, 520 prompts
* OR-Bench          bench-llm/or-bench, hard-1k and toxic splits
* XSTest            natolambert/xstest-v2-copy, over-refusal probe (250)
* StrongREJECT      jbench/strong-reject (313 prompts, with grader)
* SORRY-Bench       sorry-bench/sorry-bench-202503
* AIR-Bench         stanford-crfm/air-bench-2024

The actual ``datasets.load_dataset`` call is lazy: importing this module
doesn't pull HF datasets.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


# Each loader takes a config dict (e.g. {"dataset": "...", "max_prompts": 200})
# and returns a list of prompt dicts.
LoaderFn = Callable[[Dict[str, Any]], List[Dict[str, Any]]]


def load_harmbench(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Walledai/HarmBench standard split."""
    from datasets import load_dataset

    dataset = cfg.get("dataset", "walledai/HarmBench")
    split = cfg.get("split", "train")
    cfg_name = cfg.get("config", "standard")
    ds = load_dataset(dataset, cfg_name, split=split)
    max_p = cfg.get("max_prompts")
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if max_p is not None and idx >= max_p:
            break
        rows.append(
            {
                "source": "harmbench",
                "behavior_id": str(idx),
                "prompt": item.get("prompt", ""),
                "harmbench_category": item.get("category", ""),
            }
        )
    return rows


def load_advbench(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Walledai/AdvBench harmful-behaviors set (520 prompts)."""
    from datasets import load_dataset

    ds = load_dataset(cfg.get("dataset", "walledai/AdvBench"), split=cfg.get("split", "train"))
    max_p = cfg.get("max_prompts")
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if max_p is not None and idx >= max_p:
            break
        rows.append(
            {
                "source": "advbench",
                "behavior_id": str(idx),
                "prompt": item.get("prompt", ""),
                "advbench_target": item.get("target", ""),
            }
        )
    return rows


def load_xstest(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """XSTest over-refusal probe (250 prompts: safe + unsafe contrast)."""
    from datasets import load_dataset

    ds = load_dataset(cfg.get("dataset", "natolambert/xstest-v2-copy"),
                      split=cfg.get("split", "gpt4"))
    max_p = cfg.get("max_prompts")
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if max_p is not None and idx >= max_p:
            break
        rows.append(
            {
                "source": "xstest",
                "behavior_id": str(item.get("id", idx)),
                "prompt": item.get("prompt", ""),
                "xstest_type": item.get("type", ""),
                "expected": item.get("final_label", ""),
            }
        )
    return rows


def load_strongreject(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """StrongREJECT forbidden-prompt set (313 prompts, automated grader)."""
    from datasets import load_dataset

    ds = load_dataset(cfg.get("dataset", "walledai/StrongREJECT"), split=cfg.get("split", "train"))
    max_p = cfg.get("max_prompts")
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if max_p is not None and idx >= max_p:
            break
        rows.append(
            {
                "source": "strongreject",
                "behavior_id": str(idx),
                "prompt": item.get("forbidden_prompt", item.get("prompt", "")),
                "strongreject_category": item.get("category", ""),
                "strongreject_source": item.get("source", ""),
            }
        )
    return rows


def load_sorrybench(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sorry-Bench v1 (base prompt-style subset, ~440 prompts)."""
    from datasets import load_dataset

    ds = load_dataset(cfg.get("dataset", "sorry-bench/sorry-bench-202503"),
                      split=cfg.get("split", "train"))
    keep_style = cfg.get("filter_prompt_style", "base")
    max_p = cfg.get("max_prompts")
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if max_p is not None and len(rows) >= max_p:
            break
        if item.get("prompt_style") != keep_style:
            continue
        turns = item.get("turns")
        prompt = turns[0] if isinstance(turns, list) and turns else (
            item.get("prompt") or item.get("instruction") or ""
        )
        if not prompt:
            continue
        rows.append(
            {
                "source": "sorrybench",
                "behavior_id": item.get("question_id", item.get("id", str(idx))),
                "prompt": prompt,
                "sorrybench_category": item.get("category", ""),
            }
        )
    return rows


def load_orbench(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """OR-Bench over-refusal benchmark (hard-1k + toxic splits)."""
    from datasets import load_dataset

    splits = cfg.get("splits", ["or-bench-hard-1k", "or-bench-toxic"])
    rows: List[Dict[str, Any]] = []
    max_p = cfg.get("max_prompts")
    for split_name in splits:
        ds = load_dataset(cfg.get("dataset", "bench-llm/or-bench"), split_name,
                          split=cfg.get("split", "train"))
        for idx, item in enumerate(ds):
            if max_p is not None and len(rows) >= max_p:
                break
            rows.append(
                {
                    "source": "orbench",
                    "orbench_split": split_name,
                    "behavior_id": f"{split_name}__{idx}",
                    "prompt": item.get("prompt", ""),
                    "orbench_category": item.get("category", ""),
                }
            )
    return rows


def load_wildjailbreak(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """WildJailbreak adversarial-harmful subset (default 500 prompts).

    Canonical id: ``allenai/wildjailbreak`` (Jiang et al., arXiv:2406.18510).
    The ``eval`` config (split ``train``, 2.2K rows = 2000 adversarial_harmful
    + 210 adversarial_benign) is exported with three columns: ``adversarial``
    (the prompt string), ``label`` (int64) and ``data_type`` (string, one of
    ``adversarial_harmful`` / ``adversarial_benign``). Verified against the
    dataset viewer: ``label == 1`` rows are exactly the ``adversarial_harmful``
    ones. We filter on ``data_type`` when present (authoritative) and fall back
    to ``label == 1``. Note the dataset is gated, so loading requires an
    accepted HF licence / auth token.
    """
    from datasets import load_dataset

    max_p = cfg.get("max_prompts", 500)
    ds = load_dataset(cfg.get("dataset", "allenai/wildjailbreak"),
                      cfg.get("config", "eval"), split=cfg.get("split", "train"))
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        data_type = item.get("data_type")
        if data_type is not None:
            if data_type != "adversarial_harmful":
                continue
        elif item.get("label", 0) != 1:
            continue
        rows.append(
            {
                "source": "wildjailbreak",
                "behavior_id": str(idx),
                "prompt": item.get("adversarial", ""),
                "wildjailbreak_type": item.get("data_type", ""),
            }
        )
        if max_p and len(rows) >= max_p:
            break
    return rows


def load_agentharm(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """AgentHarm benchmark (Andriushchenko et al., ICLR 2025, arXiv:2410.09024).

    Agent-style malicious tasks. Canonical id: ``ai-safety-institute/AgentHarm``.
    The ``harmful`` config (split ``test_public``, 176 rows; ``validation`` has
    32) is verified against the dataset viewer to carry 9 columns: ``id``,
    ``id_original``, ``detailed_prompt`` (bool), ``hint_included`` (bool),
    ``name``, ``category``, ``prompt``, ``target_functions`` (list[str]),
    ``grading_function``. We map ``id`` -> ``behavior_id``, ``prompt``,
    ``category``, ``name`` and the ``hint_included`` flag.
    """
    from datasets import load_dataset

    ds = load_dataset(cfg.get("dataset", "ai-safety-institute/AgentHarm"),
                      cfg.get("config", "harmful"), split=cfg.get("split", "test_public"))
    max_p = cfg.get("max_prompts")
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if max_p is not None and idx >= max_p:
            break
        rows.append(
            {
                "source": "agentharm",
                "behavior_id": str(item.get("id", idx)),
                "prompt": item.get("prompt", ""),
                "agentharm_category": item.get("category", ""),
                "agentharm_name": item.get("name", ""),
                "agentharm_hint_included": item.get("hint_included", False),
            }
        )
    return rows


def load_cares(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """CARES medical safety benchmark.

    Canonical dataset id: ``HFXM/CARES-18K`` (CARES-18K: Clinical Adversarial
    Robustness and Evaluation of Safety; arXiv:2505.11413). Single ``default``
    config with ``test``/``train`` splits. Fields: ``principle_index``,
    ``generation_model``, ``harmful_level`` (0-3), ``method`` (direct, indirect,
    obfuscate, role-play), ``base_prompt`` (the underlying medical question),
    ``prompt`` (the final, possibly adversarially-transformed prompt).
    """
    from datasets import load_dataset

    ds = load_dataset(
        cfg.get("dataset", "HFXM/CARES-18K"),
        split=cfg.get("split", "test"),
    )
    max_p = cfg.get("max_prompts")
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if max_p is not None and idx >= max_p:
            break
        prompt = item.get("prompt") or item.get("base_prompt") or ""
        if not prompt:
            continue
        rows.append(
            {
                "source": "cares",
                "behavior_id": str(idx),
                "prompt": prompt,
                "cares_base_prompt": item.get("base_prompt", ""),
                "cares_harmful_level": item.get("harmful_level", ""),
                "cares_method": item.get("method", ""),
                "cares_principle_index": item.get("principle_index", ""),
            }
        )
    return rows


def load_airbench(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """AIR-Bench 2024 (Zeng et al., ICLR 2025).

    Regulation-derived risk taxonomy. Canonical id: ``stanford-crfm/air-bench-2024``.
    The dataset has six configs (``default``, ``china``, ``us``,
    ``eu_comprehensive``, ``eu_mandatory``, ``judge_prompts``); a config name
    is mandatory. We default to ``default`` (5.7K prompts), split ``test``.
    Fields: ``cate-idx``, ``l2-name``, ``l3-name``, ``l4-name``, ``prompt``.
    """
    from datasets import load_dataset

    ds = load_dataset(cfg.get("dataset", "stanford-crfm/air-bench-2024"),
                      cfg.get("config", "default"),
                      split=cfg.get("split", "test"))
    max_p = cfg.get("max_prompts")
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if max_p is not None and idx >= max_p:
            break
        rows.append(
            {
                "source": "airbench",
                "behavior_id": str(item.get("cate-idx", idx)),
                "prompt": item.get("prompt", ""),
                "airbench_category": item.get("l3-name", ""),
                "airbench_l2_category": item.get("l2-name", ""),
                "airbench_l4_category": item.get("l4-name", ""),
            }
        )
    return rows


def load_saladbench(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """SALAD-Bench (Li et al., ACL Findings 2024).

    Hierarchical attack/defense evaluation. Canonical id: ``OpenSafetyLab/Salad-Data``.
    The ``base_set`` config (split ``train``, 21.3K rows) has fields
    ``question``, ``qid``, ``source`` and the three-level taxonomy columns
    ``1-category``, ``2-category``, ``3-category`` (hyphenated dict keys).
    """
    from datasets import load_dataset

    ds = load_dataset(cfg.get("dataset", "OpenSafetyLab/Salad-Data"),
                      cfg.get("config", "base_set"),
                      split=cfg.get("split", "train"))
    max_p = cfg.get("max_prompts")
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if max_p is not None and idx >= max_p:
            break
        rows.append(
            {
                "source": "saladbench",
                "behavior_id": str(item.get("qid", idx)),
                "prompt": item.get("question", item.get("prompt", "")),
                "saladbench_category": item.get("1-category", ""),
                "saladbench_subcategory": item.get("2-category", ""),
                "saladbench_subsubcategory": item.get("3-category", ""),
                "saladbench_source": item.get("source", ""),
            }
        )
    return rows


def load_jailbreakbench(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """JailbreakBench (Chao et al., arXiv:2404.01318).

    Community-standard 200-behavior jailbreak evaluation suite.
    Primary source: ``walledai/JailbreakBench`` on HuggingFace (verified
    dataset viewer: fields ``behavior_id``, ``Goal``, ``Target``,
    ``Category``). Falls back to the ``jailbreakbench`` pip package if the
    HF dataset is unavailable. Raises ``ImportError`` with a clear message
    if neither source is accessible.
    """
    dataset_id = cfg.get("dataset", "walledai/JailbreakBench")
    split = cfg.get("split", "train")
    max_p = cfg.get("max_prompts")

    # --- primary: HuggingFace dataset ---
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split=split)
        rows: List[Dict[str, Any]] = []
        for idx, item in enumerate(ds):
            if max_p is not None and idx >= max_p:
                break
            # The HF dataset uses capitalised field names (Goal, Target, Category).
            # Fall back to lowercase variants for robustness.
            goal = item.get("Goal") or item.get("goal") or item.get("prompt") or ""
            rows.append(
                {
                    "source": "jailbreakbench",
                    "behavior_id": str(
                        item.get("behavior_id") or item.get("BehaviorID") or idx
                    ),
                    "prompt": goal,
                    "target": item.get("Target") or item.get("target") or "",
                    "category": item.get("Category") or item.get("category") or "",
                }
            )
        return rows
    except Exception as hf_err:
        pass  # fall through to pip-package fallback

    # --- fallback: jailbreakbench pip package ---
    try:
        import jailbreakbench as jbb  # type: ignore

        dataset = jbb.read_dataset()
        rows = []
        for idx, item in enumerate(dataset.behaviors):
            if max_p is not None and idx >= max_p:
                break
            rows.append(
                {
                    "source": "jailbreakbench",
                    "behavior_id": str(
                        getattr(item, "behavior_id", None) or idx
                    ),
                    "prompt": str(getattr(item, "goal", "") or ""),
                    "target": str(getattr(item, "target", "") or ""),
                    "category": str(getattr(item, "category", "") or ""),
                }
            )
        return rows
    except ImportError:
        pass

    raise ImportError(
        "JailbreakBench data is not available. Install via one of:\n"
        "  pip install jailbreakbench\n"
        "or ensure the HuggingFace dataset 'walledai/JailbreakBench' is "
        "accessible (may require HF authentication)."
    )


def load_muse(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """MUSE — Machine Unlearning Six-Way Evaluation (Shi et al., arXiv:2407.06460).

    Evaluates unlearning across six axes (verbatim memorization, knowledge
    memorization, privacy leakage, utility, fluency, factuality) in two
    domains. Canonical HF ids:
    * ``muse-bench/MUSE-News``  (news domain)
    * ``muse-bench/MUSE-Books`` (books domain)

    Each dataset has ``forget`` and ``retain`` splits (the split name is
    passed as the HF split argument). Returns text chunks with their
    split label and domain so downstream metrics can compute the full
    six-axis unlearning profile.

    Parameters (via cfg):
        domain (str): ``"news"`` (default) or ``"books"``.
        config (str): dataset config name. MUSE datasets are split into
            evaluation axes — pick one of: ``"verbmem"`` (default),
            ``"knowmem"``, ``"privleak"``, ``"raw"``, ``"scal"``,
            ``"sust"``, ``"train"``.
        split (str):  HF split to load. Available splits depend on the
            chosen config — for most configs the only split is
            ``"forget"`` (default).
        max_prompts (int): optional cap on returned rows.
    """
    domain = cfg.get("domain", "news").lower()
    if domain not in ("news", "books"):
        raise ValueError(f"MUSE domain must be 'news' or 'books', got {domain!r}")

    dataset_map = {
        "news": "muse-bench/MUSE-News",
        "books": "muse-bench/MUSE-Books",
    }
    dataset_id = cfg.get("dataset", dataset_map[domain])
    # MUSE datasets require a config name (evaluation axis).
    # Default to "verbmem" (verbatim memorization) — the most commonly cited axis.
    muse_config = cfg.get("config", "verbmem")
    split = cfg.get("split", "forget")
    max_p = cfg.get("max_prompts")

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required: pip install datasets"
        ) from e

    try:
        ds = load_dataset(dataset_id, muse_config, split=split)
    except Exception as e:
        raise ImportError(
            f"Could not load MUSE dataset '{dataset_id}' (split='{split}'). "
            "Ensure the dataset is accessible on HuggingFace. "
            f"Original error: {e}"
        ) from e

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if max_p is not None and idx >= max_p:
            break
        # The dataset stores the text under 'input' or 'text' depending on version.
        text = item.get("input") or item.get("text") or item.get("passage") or ""
        rows.append(
            {
                "source": "muse",
                "behavior_id": str(idx),
                "text": text,
                "split": split,
                "domain": domain,
            }
        )
    return rows


def load_rwku(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """RWKU — Real-World Knowledge Unlearning (Jin et al., arXiv:2406.10890).

    Benchmark for unlearning 200 public-figure entities from LLMs.
    Canonical HF id: ``jinzhuoran/RWKU``.

    The dataset has a forget set (knowledge about 200 public figures) and a
    retain set. Fields in the HF dataset: ``question``, ``answer``,
    ``subject`` (the public-figure name), and the split is identified by the
    HF split argument (``forget`` / ``retain`` / ``test``).

    Parameters (via cfg):
        config (str): dataset config / axis. Default ``"forget_target"``
            (the 200 forget-set entities with QA pairs). Other options:
            ``"forget_level1/2/3"``, ``"neighbor_level1/2"``,
            ``"mia_forget"``, ``"mia_retain"``,
            ``"utility_general/reason/truthfulness/factuality/fluency"``.
        split (str): HF split — most configs have only ``"train"`` (default).
        max_prompts (int): optional cap.
    """
    dataset_id = cfg.get("dataset", "jinzhuoran/RWKU")
    rwku_config = cfg.get("config", "forget_target")
    split = cfg.get("split", "train")
    max_p = cfg.get("max_prompts")

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required: pip install datasets"
        ) from e

    try:
        ds = load_dataset(dataset_id, rwku_config, split=split)
    except Exception as e:
        raise ImportError(
            f"Could not load RWKU dataset '{dataset_id}' (split='{split}'). "
            "Ensure the dataset is accessible on HuggingFace. "
            f"Original error: {e}"
        ) from e

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if max_p is not None and idx >= max_p:
            break
        rows.append(
            {
                "source": "rwku",
                "behavior_id": str(item.get("id", idx)),
                "prompt": item.get("question", ""),
                "answer": item.get("answer", ""),
                "subject": item.get("subject", item.get("entity", "")),
                "split": split,
            }
        )
    return rows


def load_safedialbench(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """SafeDialBench — Multi-turn Safety Benchmark (Zhao et al., arXiv:2502.11090).

    Covers 22 safety scenarios × 7 jailbreak attack types across multi-turn
    conversations. Canonical HF id: ``thu-coai/SafeDialBench``.

    Each example is a multi-turn conversation labelled safe or unsafe.
    This loader normalises each example into a dict with a ``conversation``
    field (list of ``{"role": str, "content": str}`` dicts) plus metadata.

    Parameters (via cfg):
        split (str): HF split to load — ``"test"`` (default).
        max_prompts (int): optional cap on returned examples.
    """
    # NOTE: As of 2026-05-16 this dataset has not been released publicly on
    # HuggingFace Hub (thu-coai/SafeDialBench returns 404). The loader is
    # wired up and ready; update the default dataset_id below once the
    # authors publish it, or pass cfg['dataset'] with the correct id.
    dataset_id = cfg.get("dataset", "thu-coai/SafeDialBench")
    split = cfg.get("split", "test")
    max_p = cfg.get("max_prompts")

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required: pip install datasets"
        ) from e

    try:
        ds = load_dataset(dataset_id, split=split)
    except Exception as e:
        raise ImportError(
            f"Could not load SafeDialBench dataset '{dataset_id}' "
            f"(split='{split}'). "
            "Try: pip install datasets && huggingface-cli login\n"
            "If the dataset id has changed, pass cfg['dataset'] explicitly.\n"
            f"Original error: {e}"
        ) from e

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if max_p is not None and idx >= max_p:
            break

        # Normalise the conversation field. The dataset may store it as:
        # (a) a list of {"role": ..., "content": ...} dicts (standard),
        # (b) a list of strings alternating user/assistant,
        # (c) a single string of the full dialogue.
        raw_conv = item.get("conversation") or item.get("dialogue") or item.get("messages") or []
        conversation: List[Dict[str, str]]
        if isinstance(raw_conv, list) and raw_conv:
            first = raw_conv[0]
            if isinstance(first, dict):
                # Already in the standard format; ensure keys are present.
                conversation = [
                    {
                        "role": str(turn.get("role", "user")),
                        "content": str(turn.get("content", turn.get("text", ""))),
                    }
                    for turn in raw_conv
                ]
            else:
                # List of plain strings — interleave user/assistant roles.
                roles = ["user", "assistant"]
                conversation = [
                    {"role": roles[i % 2], "content": str(turn)}
                    for i, turn in enumerate(raw_conv)
                ]
        elif isinstance(raw_conv, str):
            conversation = [{"role": "user", "content": raw_conv}]
        else:
            conversation = []

        rows.append(
            {
                "source": "safedialbench",
                "behavior_id": str(item.get("id", idx)),
                "conversation": conversation,
                "category": item.get("category") or item.get("scenario") or "",
                "attack_type": item.get("attack_type") or item.get("jailbreak_type") or "",
                "label": item.get("label") or item.get("safety_label") or "",
            }
        )
    return rows


LOADERS: Dict[str, LoaderFn] = {
    "harmbench": load_harmbench,
    "advbench": load_advbench,
    "xstest": load_xstest,
    "strongreject": load_strongreject,
    "sorrybench": load_sorrybench,
    "orbench": load_orbench,
    "wildjailbreak": load_wildjailbreak,
    "agentharm": load_agentharm,
    "cares": load_cares,
    "airbench": load_airbench,
    "saladbench": load_saladbench,
    "jailbreakbench": load_jailbreakbench,
    "muse": load_muse,
    "rwku": load_rwku,
    "safedialbench": load_safedialbench,
}


def load_prompts(name: str, cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Public entrypoint: dispatch by benchmark name."""
    if name not in LOADERS:
        raise ValueError(f"Unknown benchmark: {name!r}. Known: {sorted(LOADERS)}")
    return LOADERS[name](cfg or {})


__all__ = ["LOADERS", "load_prompts"]
