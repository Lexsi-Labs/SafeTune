"""
Capability evaluation via EleutherAI's ``lm-evaluation-harness``.

This wraps the lm-eval-harness Python API (not the CLI) so SafeTune library
users can invoke standard utility benchmarks (GSM8K, MMLU, TruthfulQA,
HellaSwag, IFEval, MBPP) on a model or a steered model without shelling out.

Two model paths supported:

* ``backend="hf"``: use HF ``transformers`` via lm-eval's ``HFLM`` wrapper.
  Works for any HF id or local path.
* ``backend="vllm"``: use ``vLLM`` for batched throughput.

Both modes go through ``lm_eval.simple_evaluate`` and return the same shape:

    {
        "gsm8k": {"acc": 0.42, "acc_stderr": 0.01, ...},
        "mmlu": {"acc": 0.38, ...},
        ...
    }

Mirrors the cthetha-eval ``run_lmeval_rebuttal.sh`` workflow but as a Python
function call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# A practical default task set chosen for breadth across capability axes.
DEFAULT_TASK_SETS: Dict[str, List[str]] = {
    "core": ["gsm8k", "mmlu", "truthfulqa_mc2", "hellaswag"],
    "math": ["gsm8k", "gsm8k_cot", "mathqa"],
    "reasoning": ["mmlu", "hellaswag", "winogrande", "arc_challenge"],
    "code": ["mbpp", "humaneval"],
    "safety_adjacent": ["truthfulqa_mc2", "truthfulqa_gen", "toxigen"],
    "ifeval": ["ifeval"],
    # EMNLP 2026 paper grid (4 domains + domain-agnostic).
    # Maps directly to docs/EMNLP_PLAN.md "Capability benchmarks (5)" table.
    "emnlp_math": ["gsm8k", "hendrycks_math"],
    "emnlp_code": ["mbpp", "humaneval"],
    "emnlp_medical": ["medqa_4options", "pubmedqa"],
    "emnlp_legal": ["legalbench"],
    "emnlp_general": ["mmlu"],
    "emnlp_all": [
        "gsm8k", "hendrycks_math",
        "mbpp", "humaneval",
        "medqa_4options", "pubmedqa",
        "legalbench",
        "mmlu",
    ],
}


@dataclass
class CapabilityEvalConfig:
    """Configuration for a capability eval run.

    Attributes:
        tasks: list of lm-eval task names. Convenience: pass one of the keys
            in ``DEFAULT_TASK_SETS`` (e.g. ``"core"``) to expand to that set.
        backend: ``"hf"`` or ``"vllm"``.
        model: HF id or local path. For ``"hf"`` you may also pass a
            pre-instantiated HFLM via ``hflm`` instead.
        hflm: optional pre-built lm-eval HFLM (skips re-loading the model).
        device: ``"cuda"``, ``"cuda:1"``, ``"cpu"``, ``"auto"``.
        batch_size: lm-eval batch size. ``"auto"`` is also accepted by lm-eval.
        limit: cap on examples per task (useful for smoke tests).
        num_fewshot: per-task default; can be overridden per-task via lm-eval.
        max_length: tokenizer max length (only used by ``"hf"`` backend).
        trust_remote_code: passed to lm-eval.
        log_samples: store full per-sample outputs in the result dict.
        output_path: if set, write a JSON dump of the full lm-eval result there.
    """

    tasks: List[str] = field(default_factory=lambda: ["gsm8k", "mmlu"])
    backend: str = "hf"
    model: Optional[str] = None
    hflm: Optional[Any] = None
    device: str = "auto"
    batch_size: Any = 8
    limit: Optional[int] = None
    num_fewshot: Optional[int] = None
    max_length: Optional[int] = None
    trust_remote_code: bool = True
    log_samples: bool = False
    output_path: Optional[str] = None
    extra_model_args: Dict[str, Any] = field(default_factory=dict)


def _expand_task_set(name_or_list: List[str]) -> List[str]:
    out: List[str] = []
    for n in name_or_list:
        if n in DEFAULT_TASK_SETS:
            out.extend(DEFAULT_TASK_SETS[n])
        else:
            out.append(n)
    # dedupe preserving order
    seen = set()
    deduped: List[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def _build_hflm(cfg: CapabilityEvalConfig) -> Any:
    """Construct an lm-eval ``HFLM`` for ``cfg.model``."""
    if cfg.hflm is not None:
        return cfg.hflm
    if not cfg.model:
        raise ValueError("CapabilityEvalConfig: ``model`` required when ``hflm`` is None.")
    from lm_eval.models.huggingface import HFLM

    kwargs: Dict[str, Any] = {
        "pretrained": cfg.model,
        "device": cfg.device,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.max_length is not None:
        kwargs["max_length"] = cfg.max_length
    kwargs.update(cfg.extra_model_args)
    logger.info("CapabilityEval: loading HFLM(%s)", cfg.model)
    return HFLM(**kwargs)


def _build_vllm(cfg: CapabilityEvalConfig) -> Any:
    """Construct an lm-eval ``VLLM`` model wrapper."""
    if not cfg.model:
        raise ValueError("CapabilityEvalConfig: ``model`` required for vllm backend.")
    from lm_eval.models.vllm_causallms import VLLM

    kwargs: Dict[str, Any] = {
        "pretrained": cfg.model,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.max_length is not None:
        kwargs["max_model_len"] = cfg.max_length
    kwargs.update(cfg.extra_model_args)
    logger.info("CapabilityEval: loading VLLM(%s)", cfg.model)
    return VLLM(**kwargs)


def run_capability_eval(config: CapabilityEvalConfig) -> Dict[str, Dict[str, Any]]:
    """Run the configured tasks via lm-evaluation-harness.

    Returns a dict mapping task name to its metrics dict.
    """
    try:
        from lm_eval import simple_evaluate
    except ImportError as e:
        raise ImportError(
            "run_capability_eval requires lm-evaluation-harness. "
            "Install with `pip install lm-eval` and retry."
        ) from e

    tasks = _expand_task_set(config.tasks)
    if not tasks:
        raise ValueError("CapabilityEvalConfig: empty task list.")

    backend = config.backend.lower()
    if backend == "hf":
        lm = _build_hflm(config)
    elif backend == "vllm":
        lm = _build_vllm(config)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    se_kwargs: Dict[str, Any] = {
        "model": lm,
        "tasks": tasks,
        "batch_size": config.batch_size,
        "log_samples": config.log_samples,
    }
    if config.limit is not None:
        se_kwargs["limit"] = config.limit
    if config.num_fewshot is not None:
        se_kwargs["num_fewshot"] = config.num_fewshot

    logger.info("CapabilityEval: simple_evaluate(tasks=%s, limit=%s)", tasks, config.limit)
    result = simple_evaluate(**se_kwargs)

    if config.output_path:
        import json as _json
        from pathlib import Path

        out_p = Path(config.output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # ``samples`` can be huge; only persist them when explicitly requested.
        payload: Dict[str, Any] = {"results": result.get("results", {}), "configs": result.get("configs", {})}
        if config.log_samples and "samples" in result:
            payload["samples"] = result["samples"]
        out_p.write_text(_json.dumps(payload, indent=2, default=str))
        logger.info("CapabilityEval: wrote results to %s", out_p)

    return result.get("results", {})


__all__ = [
    "CapabilityEvalConfig",
    "DEFAULT_TASK_SETS",
    "run_capability_eval",
]
