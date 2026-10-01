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

Benchmarks that ``evaluate()`` also runs (HarmBench, AdvBench, XSTest,
SORRY-Bench, OR-Bench, WildJailbreak, JailbreakBench, MUSE, RWKU,
SafeDialBench) are loaded by their one loader in
:mod:`safetune.data.loaders.benchmarks`; the functions here only shape the rows.
WildJailbreak keeps its load_prompts() defaults: adversarial_harmful rows only,
at most 500.

The actual ``datasets.load_dataset`` call is lazy: importing this module
doesn't pull HF datasets.

Default sources, configs and splits come from :mod:`safetune.data.dataset_ids`
(overridable with ``safetune.configure(datasets=...)``); ``cfg`` keys
``dataset`` / ``config`` / ``split`` override them per call.
"""
from __future__ import annotations

from collections import Counter
from itertools import islice
from typing import Any, Callable, Dict, List, Optional

from safetune.data.dataset_ids import load, spec
from safetune.data.loaders import benchmarks as B


def _load(name: str, cfg: Dict[str, Any], **extra):
    """``dataset_ids.load(name)`` with the ``cfg`` per-call overrides applied."""
    return load(name, source=cfg.get("dataset"), config=cfg.get("config"),
                split=cfg.get("split"), **extra)


def _spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """``cfg``'s per-call overrides as dataset-table fields for the shared loaders."""
    s = {"source": cfg.get("dataset"), "config": cfg.get("config"), "split": cfg.get("split")}
    return {k: v for k, v in s.items() if v is not None}


def _take(rows, cfg: Dict[str, Any]):
    """The first ``cfg["max_prompts"]`` rows (all when unset)."""
    return islice(rows, cfg.get("max_prompts"))


# Each loader takes a config dict (e.g. {"dataset": "...", "max_prompts": 200})
# and returns a list of prompt dicts.
LoaderFn = Callable[[Dict[str, Any]], List[Dict[str, Any]]]


def load_harmbench(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Walledai/HarmBench standard split."""
    return [
        {"source": "harmbench", "behavior_id": str(idx), "prompt": item.get("prompt", ""),
         "harmbench_category": item.get("category", "")}
        for idx, item in enumerate(_take(B.load_harmbench(**_spec(cfg)), cfg))
    ]


def load_advbench(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Walledai/AdvBench harmful-behaviors set (520 prompts)."""
    return [
        {"source": "advbench", "behavior_id": str(idx), "prompt": item.get("prompt", ""),
         "advbench_target": item.get("target", "")}
        for idx, item in enumerate(_take(B.load_advbench(**_spec(cfg)), cfg))
    ]


def load_xstest(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """XSTest over-refusal probe (walledai/XSTest: 250 safe + 200 unsafe contrast)."""
    return [
        {"source": "xstest", "behavior_id": str(item.get("id", idx)),
         "prompt": item.get("prompt", ""), "xstest_type": item.get("type", ""),
         "expected": item.get("final_label", item.get("label", ""))}
        for idx, item in enumerate(_take(B.load_xstest(**_spec(cfg)), cfg))
    ]


def load_strongreject(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """StrongREJECT forbidden-prompt set (313 prompts, automated grader)."""
    ds = _load("strongreject", cfg)
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
    # The prompt-style filter is the table's ``where`` (default {"prompt_style": "base"});
    # cfg["filter_prompt_style"] overrides it for this call.
    style = cfg.get("filter_prompt_style")
    extra = {"where": {"prompt_style": style}} if style else {}
    rows = (
        {"source": "sorrybench",
         "behavior_id": item.get("question_id", item.get("id", str(idx))),
         "prompt": item.get("prompt") or item.get("instruction") or "",
         "sorrybench_category": item.get("category", "")}
        for idx, item in enumerate(B.load_sorrybench(**_spec(cfg), **extra))
    )
    return list(_take((r for r in rows if r["prompt"]), cfg))


def load_orbench(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """OR-Bench over-refusal benchmark (hard-1k + toxic splits)."""
    seen: Counter = Counter()
    rows: List[Dict[str, Any]] = []
    for item in _take(B.load_orbench(cfg.get("splits"), **_spec(cfg)), cfg):
        split_name = item["split"] or None
        rows.append(
            {
                "source": "orbench",
                "orbench_split": split_name,
                "behavior_id": f"{split_name}__{seen[split_name]}",
                "prompt": item.get("prompt", ""),
                "orbench_category": item.get("category", ""),
            }
        )
        seen[split_name] += 1
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
    rows = (
        {"source": "wildjailbreak", "behavior_id": str(idx), "prompt": item.get("prompt", ""),
         "wildjailbreak_type": item.get("data_type", "")}
        for idx, item in enumerate(B.load_wildjailbreak(**_spec(cfg)))
        if (item.get("data_type") == "adversarial_harmful" if item.get("data_type") is not None
            else item.get("label", 0) == 1)
    )
    return list(islice(rows, cfg.get("max_prompts", 500) or None))


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
    ds = _load("agentharm", cfg)
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
    ds = _load("cares", cfg)
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
    ds = _load("airbench", cfg)
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
    ds = _load("saladbench", cfg)
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
    """JailbreakBench (Chao et al., arXiv:2404.01318): the 100 harmful
    JBB-Behaviors (``JailbreakBench/JBB-Behaviors``, ``behaviors``/``harmful``)."""
    return [
        {"source": "jailbreakbench", "behavior_id": str(item.get("Index", idx)),
         "prompt": item.get("prompt", ""), "target": item.get("Target", ""),
         "category": item.get("Category", "")}
        for idx, item in enumerate(_take(B.load_jailbreakbench(**_spec(cfg)), cfg))
    ]


def load_muse(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """MUSE (Shi et al., arXiv:2407.06460): ``cfg["domain"]`` ``"news"`` (default)
    or ``"books"``; default config ``verbmem``, split ``forget``. The passage is
    in both ``prompt`` and ``text``."""
    domain = cfg.get("domain", "news").lower()
    ds = B.load_muse(domain, **_spec(cfg))
    split = cfg.get("split") or spec(f"muse_{domain}").get("split")
    return [
        {"source": "muse", "behavior_id": str(idx), "prompt": item.get("prompt", ""),
         "text": item.get("prompt", ""), "split": split, "domain": domain}
        for idx, item in enumerate(_take(ds, cfg))
    ]


def load_rwku(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """RWKU (Jin et al., arXiv:2406.10890): forget-set QA probes (default config
    ``forget_level2``, split ``test``)."""
    split = cfg.get("split") or spec("rwku").get("split")
    return [
        {"source": "rwku", "behavior_id": str(item.get("id", idx)),
         "prompt": item.get("prompt", ""), "answer": item.get("answer", ""),
         "subject": item.get("subject", item.get("entity", "")), "split": split}
        for idx, item in enumerate(_take(B.load_rwku(**_spec(cfg)), cfg))
    ]


def load_safedialbench(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """SafeDialBench (Zhao et al., arXiv:2502.11090): multi-turn dialogues, each
    with a ``conversation`` and a flattened single-turn ``prompt``. Loads from
    ``HongyeCao/SafeDialBench``; see
    :func:`safetune.data.loaders.benchmarks.load_safedialbench`."""
    return [
        {"source": "safedialbench", "behavior_id": str(item.get("id", idx)),
         "conversation": item["conversation"], "prompt": item["prompt"],
         "category": item.get("category") or item.get("scenario") or "",
         "attack_type": item.get("attack_type") or item.get("jailbreak_type") or "",
         "label": item.get("label") or item.get("safety_label") or ""}
        for idx, item in enumerate(_take(B.load_safedialbench(**_spec(cfg)), cfg))
    ]


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
