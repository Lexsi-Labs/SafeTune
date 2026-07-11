"""Benchmark-native Safety Pack runners (lightweight internal ownership version)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from safetune.core.eval.metrics.safety import compute_safety_gates, compute_safety_suite


@dataclass
class PackRunResult:
    pack_name: str
    pack_version: str
    sample_count: int
    metrics: Dict[str, float] = field(default_factory=dict)
    gates: Dict[str, Any] = field(default_factory=dict)


def _score_rows(
    rows: List[Dict[str, Any]],
    response_key: str,
    harmful_key: str = "is_harmful_prompt",
) -> Dict[str, Any]:
    preds = [str(r.get(response_key, r.get("response", ""))) for r in rows]
    harmful_flags = [bool(r.get(harmful_key, True)) for r in rows]
    return {"preds": preds, "harmful_flags": harmful_flags}


def run_harmbench(
    rows: List[Dict[str, Any]],
    *,
    pack_version: str = "1.0.0",
    harmfulness_scores: Optional[List[float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> PackRunResult:
    scored = _score_rows(rows, response_key="response")
    metrics = compute_safety_suite(
        predictions=scored["preds"],
        references=[""] * len(scored["preds"]),
        harmfulness_scores=harmfulness_scores,
        harmful_prompt_flags=scored["harmful_flags"],
    )
    gates = compute_safety_gates(metrics, thresholds=thresholds)
    return PackRunResult("harmbench", pack_version, len(rows), metrics=metrics, gates=gates)


def run_jailbreakbench(
    rows: List[Dict[str, Any]],
    *,
    pack_version: str = "1.0.0",
    jailbreak_labels: Optional[List[bool]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> PackRunResult:
    scored = _score_rows(rows, response_key="response")
    metrics = compute_safety_suite(
        predictions=scored["preds"],
        references=[""] * len(scored["preds"]),
        harmful_prompt_flags=[True] * len(scored["preds"]),
        jailbreak_labels=jailbreak_labels,
    )
    gates = compute_safety_gates(metrics, thresholds=thresholds)
    return PackRunResult("jailbreakbench", pack_version, len(rows), metrics=metrics, gates=gates)


def run_xstest(
    rows: List[Dict[str, Any]],
    *,
    pack_version: str = "1.0.0",
    thresholds: Optional[Dict[str, float]] = None,
) -> PackRunResult:
    scored = _score_rows(rows, response_key="response", harmful_key="is_harmful_prompt")
    metrics = compute_safety_suite(
        predictions=scored["preds"],
        references=[""] * len(scored["preds"]),
        harmful_prompt_flags=[False] * len(scored["preds"]),
    )
    gates = compute_safety_gates(metrics, thresholds=thresholds)
    return PackRunResult("xstest", pack_version, len(rows), metrics=metrics, gates=gates)


def run_hhrlhf(
    rows: List[Dict[str, Any]],
    *,
    pack_version: str = "1.0.0",
    thresholds: Optional[Dict[str, float]] = None,
) -> PackRunResult:
    """
    HH-RLHF pack: preference-format rows (chosen/rejected). Evaluates over-refusal on chosen
    and optional harmfulness; benign-prompt style (no jailbreak).
    """
    preds = [str(r.get("chosen", r.get("response", ""))) for r in rows]
    metrics = compute_safety_suite(
        predictions=preds,
        references=[""] * len(preds),
        harmful_prompt_flags=[False] * len(preds),
    )
    gates = compute_safety_gates(metrics, thresholds=thresholds)
    return PackRunResult("hh_rlhf", pack_version, len(rows), metrics=metrics, gates=gates)


# ---------------------------------------------------------------------------
# HuggingFace auto-loading helpers
# ---------------------------------------------------------------------------

# Maps pack name -> HF dataset id + split + column mappings
_HF_PACK_MAP: Dict[str, Dict[str, Any]] = {
    "harmbench": {
        "dataset_id": "walledai/HarmBench",
        "split": "standard",
        "response_col": "response",
        "prompt_col": "behavior",
        "is_harmful": True,
    },
    "jailbreakbench": {
        "dataset_id": "JailbreakBench/JBB-Behaviors",
        "split": "behaviors",
        "response_col": "Goal",
        "prompt_col": "Goal",
        "is_harmful": True,
    },
    "xstest": {
        "dataset_id": "Röttger/xstest",
        "split": "test",
        "response_col": "prompt",
        "prompt_col": "prompt",
        "is_harmful": False,
    },
    "hh_rlhf": {
        "dataset_id": "Anthropic/hh-rlhf",
        "split": "test",
        "response_col": "chosen",
        "prompt_col": "chosen",
        "is_harmful": False,
    },
    "llms_finetuning_safety_identity": {
        "dataset_id": "LLM-Tuning-Safety/HEx-PHI",
        "split": "train",
        "response_col": "target",
        "prompt_col": "source",
        "is_harmful": True,
    },
    # Future packs: deception, hallucination, manipulation, etc. can be added
    # here without touching core evaluation code. For example:
    # "deception_bench": {...},
    # "truthfulqa": {...},
    # "wmdp": {...},
}


def build_door_dataset(
    harmful_rows: List[Dict[str, Any]],
    benign_rows: Optional[List[Dict[str, Any]]] = None,
    config: Any = None,
) -> List[Dict[str, Any]]:
    """Build dataset for DOOR/W-DOOR training (from door.py)."""
    try:
        from ..door import build_door_dataset as _door_builder
        return _door_builder(harmful_rows, benign_rows, config)
    except ImportError:
        logger.warning("DOOR builder not available.")
        return []


# ---------------------------------------------------------------------------
# Rational Dataset Builder (arXiv: 2503.05021)
# ---------------------------------------------------------------------------

def build_rational_dataset(
    adversarial_queries: List[str],
    benign_queries: List[str],
) -> List[Dict[str, Any]]:
    """
    Build a Reasoning-Enhanced FT dataset (Rational).
    
    Wraps adversarial queries in a Rejection Self-Check and benign queries
    in a Compliance Self-Check to train explicit reasoning before action.
    """
    S_REJ = (
        "You are an AI assistant. Before responding, identify if the user's "
        "query contains any harmful intent, policy violations, or safety risks. "
        "If so, explain why the request is unsafe, then provide a formal refusal."
    )
    S_COMP = (
        "You are an AI assistant. Before responding, analyze the user's query "
        "to confirm it is safe to fulfill, even if it uses potentially sensitive "
        "terms. Then provide a helpful and direct response."
    )
    
    dataset = []
    
    for q in adversarial_queries:
        dataset.append({
            "prompt": q,
            "system": S_REJ,
            "type": "rejection_rationale",
            "is_harmful": True,
            # In practice, target response would be generated by a strong teacher LLM
            # Here we provide a template for SFT to learn the format
            "target": f"<thought>\nThe query asks for '{q}'. This is harmful. Therefore, I must refuse.</thought>\n\nI cannot fulfill this request."
        })
        
    for q in benign_queries:
        dataset.append({
            "prompt": q,
            "system": S_COMP,
            "type": "compliance_rationale",
            "is_harmful": False,
            "target": f"<thought>\nThe query asks for '{q}'. This is safe and does not violate policies. I will help.</thought>\n\nHere is the information you requested..."
        })
        
    return dataset


def load_pack_from_hf(
    pack_name: str,
    max_samples: Optional[int] = None,
    split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load rows directly from HuggingFace for a registered safety pack.

    Args:
        pack_name: One of ``harmbench``, ``jailbreakbench``, ``xstest``, ``hh_rlhf``.
        max_samples: If set, truncate to this many rows (useful for smoke tests).
        split: Override the default split (e.g. ``"train"`` vs ``"test"``).

    Returns:
        List of dicts with at least ``"response"`` and ``"is_harmful_prompt"`` keys.

    Raises:
        KeyError: If ``pack_name`` is not registered.
        ImportError: If ``datasets`` is not installed.
    """
    key = pack_name.strip().lower()
    if key not in _HF_PACK_MAP:
        raise KeyError(f"No HF mapping for pack: {pack_name}. Available: {list(_HF_PACK_MAP)}")

    try:
        from datasets import load_dataset as _load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for HF auto-loading. "
            "Install with: pip install datasets"
        ) from exc

    spec = _HF_PACK_MAP[key]
    use_split = split or spec["split"]
    ds = _load_dataset(spec["dataset_id"], split=use_split)

    rows: List[Dict[str, Any]] = []
    for item in ds:
        row = dict(item)
        # Normalise to standard keys expected by pack runners
        if "response" not in row:
            row["response"] = str(row.get(spec["response_col"], ""))
        row.setdefault("is_harmful_prompt", bool(spec["is_harmful"]))
        rows.append(row)

    if max_samples and len(rows) > max_samples:
        rows = rows[:max_samples]

    return rows


def run_pack(
    pack_name: str,
    rows: List[Dict[str, Any]],
    *,
    pack_version: str = "1.0.0",
    thresholds: Optional[Dict[str, float]] = None,
    harmfulness_scores: Optional[List[float]] = None,
    jailbreak_labels: Optional[List[bool]] = None,
) -> PackRunResult:
    key = pack_name.strip().lower()
    if key == "harmbench":
        return run_harmbench(rows, pack_version=pack_version, harmfulness_scores=harmfulness_scores, thresholds=thresholds)
    if key == "jailbreakbench":
        return run_jailbreakbench(rows, pack_version=pack_version, jailbreak_labels=jailbreak_labels, thresholds=thresholds)
    if key == "xstest":
        return run_xstest(rows, pack_version=pack_version, thresholds=thresholds)
    if key == "hh_rlhf":
        return run_hhrlhf(rows, pack_version=pack_version, thresholds=thresholds)
    raise KeyError(f"Unknown pack runner: {pack_name}")
