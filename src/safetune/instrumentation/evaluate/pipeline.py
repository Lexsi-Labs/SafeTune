"""Top-level safety evaluation pipeline.

Orchestrates: data loading → response generation → judging → metrics,
matching cthetha-eval's SOURCE_TO_JUDGE routing exactly.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .generate import generate_responses
from .judges import JUDGE_REGISTRY, run_judge

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# cthetha-eval SOURCE_TO_JUDGE routing
# ---------------------------------------------------------------------------

_SOURCE_TO_JUDGE: Dict[str, str] = {
    "harmbench": "harmbench",
    "wildjailbreak": "wildguard",
    "sorrybench_v1": "sorrybench",
    "hexphi": "hexphi",
    "orbench": "orbench",
    "ailuminate": "ailuminate",
    "advbench": "advbench",
}

# OR-Bench has two sub-splits tracked separately.
_ORBENCH_SPLITS = ("or-bench-hard-1k", "or-bench-toxic")

# safetune bench name → cthetha-eval file stem (for write_bench_jsonl raw output)
_BENCH_TO_FILE: Dict[str, str] = {
    "harmbench":    "harmbench",
    "wildjailbreak": "wildjailbreak",
    "sorrybench_v1": "sorrybench_v1",
    "hexphi":       "hexphi",
    "orbench":      "orbench",
    "ailuminate":   "ailuminate",
    "advbench":     "advbench",
}

# Loader map: benchmark name → (module_path, function_name)
_LOADERS: Dict[str, tuple[str, str]] = {
    "harmbench": ("safetune.data.loaders.benchmarks", "load_harmbench"),
    "wildjailbreak": ("safetune.data.loaders.benchmarks", "load_wildjailbreak"),
    "sorrybench_v1": ("safetune.data.loaders.benchmarks", "load_sorrybench"),
    "hexphi": ("safetune.data.loaders.benchmarks", "load_hexphi"),
    "orbench": ("safetune.data.loaders.benchmarks", "load_orbench"),
    "ailuminate": ("safetune.data.loaders.benchmarks", "load_ailuminate"),
    "advbench": ("safetune.data.loaders.benchmarks", "load_advbench"),
}


def _load_benchmark(name: str) -> list:
    """Load a benchmark dataset by name via the safetune.data.loaders.benchmarks
    module. Returns a list of dicts with at least a 'prompt' key."""
    import importlib

    if name not in _LOADERS:
        raise ValueError(
            f"Unknown benchmark: {name!r}. Supported: {list(_LOADERS)}"
        )
    mod_path, fn_name = _LOADERS[name]
    loader = getattr(importlib.import_module(mod_path), fn_name)
    ds = loader()

    # Normalise to list[dict] with a 'prompt' field.
    rows: list = []
    for item in ds:
        if isinstance(item, dict):
            rows.append(item)
        else:
            rows.append({"prompt": str(item)})
    return rows


def _extract_prompts(rows: list) -> List[str]:
    prompts: List[str] = []
    for item in rows:
        p = item.get("prompt") or item.get("text") or item.get("instruction") or ""
        prompts.append(str(p))
    return prompts


def _compute_bench_metrics(scores: List[float]) -> Dict[str, Any]:
    n = len(scores)
    if n == 0:
        return {"n": 0, "asr": 0.0, "refusal_rate": 1.0, "scores": []}
    asr = sum(scores) / n
    return {
        "n": n,
        "asr": asr,
        "refusal_rate": 1.0 - asr,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# JSONL output helper
# ---------------------------------------------------------------------------

def write_jsonl(
    results: Dict[str, Dict[str, Any]],
    output_dir: str,
    model_folder_name: str,
) -> None:
    """Write per-benchmark scored JSONL files in the cthetha-eval harness format.

    Output filename pattern:
        <output_dir>/<model_folder_name>__base__<bench>_scored.jsonl

    Each line is a JSON object:
        {"prompt": ..., "response": ..., "score": ..., "bench": ...}

    The harness.py ``safety_metrics()`` function can read these files to
    recompute aggregate metrics without re-running generation or judging.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for bench_name, bench_data in results.items():
        fname = out_path / f"{model_folder_name}__base__{bench_name}_scored.jsonl"
        prompts = bench_data.get("prompts", [])
        responses = bench_data.get("responses", [])
        scores = bench_data.get("scores", [])

        if not scores:
            log.warning("No scores for %s — skipping file write.", bench_name)
            continue
        with open(fname, "w", encoding="utf-8") as fh:
            for i, score in enumerate(scores):
                record = {
                    "prompt": prompts[i] if i < len(prompts) else "",
                    "response": responses[i] if i < len(responses) else "",
                    "score": score,
                    "refused": score < 0.5,  # harness._refusal_rate() reads this field
                    "bench": bench_name,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        log.info("Wrote %d records to %s", len(scores), fname)


# ---------------------------------------------------------------------------
# Split pipeline: generate first, then score
# ---------------------------------------------------------------------------

def generate_bench_responses(
    model_path: str,
    benchmarks: List[str],
    *,
    gpu_memory_utilization: float = 0.5,
    max_new_tokens: int = 512,
    tokenizer_name: Optional[str] = None,
    backend: str = "vllm",
) -> Dict[str, Dict[str, Any]]:
    """Phase 1: load target model once, generate responses for all benchmarks.

    Returns dict: bench_name → {prompts, responses, rows}.
    Does NOT run judges — call _score_with_judges separately after GPU flush.
    """
    bench_data: Dict[str, Dict] = {}
    valid_benches = []
    for bench_name in benchmarks:
        try:
            rows = _load_benchmark(bench_name)
        except Exception as exc:
            log.error("[%s] Failed to load dataset: %s", bench_name, exc)
            continue
        prompts = _extract_prompts(rows)
        if not prompts:
            log.warning("[%s] No prompts found — skipping.", bench_name)
            continue
        bench_data[bench_name] = {"rows": rows, "prompts": prompts}
        valid_benches.append(bench_name)

    if not valid_benches:
        return {}

    all_prompts: List[str] = []
    offsets: Dict[str, tuple] = {}
    for bench_name in valid_benches:
        start = len(all_prompts)
        all_prompts.extend(bench_data[bench_name]["prompts"])
        offsets[bench_name] = (start, len(all_prompts))

    log.info("Generating for %d prompts across %d benchmarks...",
             len(all_prompts), len(valid_benches))
    try:
        all_responses = generate_responses(
            model_path,
            all_prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,
            apply_chat_template=True,
            tokenizer_name=tokenizer_name,
            backend=backend,
        )
    except Exception as exc:
        log.error("Generation failed: %s", exc)
        return {}

    for bench_name in valid_benches:
        start, end = offsets[bench_name]
        bench_data[bench_name]["responses"] = all_responses[start:end]

    return bench_data


def write_bench_jsonl(
    bench_data: Dict[str, Dict[str, Any]],
    output_dir: str,
    model_folder_name: str,
) -> None:
    """Write raw (unscored) response JSONL files for _score_with_judges to read.

    Output: <output_dir>/<model_folder_name>__base__<bench>.jsonl
    Each line: {"prompt": ..., "response": ...}
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for bench_name, data in bench_data.items():
        prompts   = data.get("prompts", [])
        responses = data.get("responses", [])
        if not responses:
            log.warning("[%s] No responses to write.", bench_name)
            continue
        # Map safetune bench name → cthetha-eval file stem
        file_stem = _BENCH_TO_FILE.get(bench_name, bench_name)
        fname = out_path / f"{model_folder_name}__base__{file_stem}.jsonl"
        with open(fname, "w", encoding="utf-8") as fh:
            for p, r in zip(prompts, responses):
                fh.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")
        log.info("Wrote %d responses to %s", len(responses), fname)


# ---------------------------------------------------------------------------
# Main pipeline (combined generate+score, kept for compatibility)
# ---------------------------------------------------------------------------

def run_safety_eval(
    model_path: str,
    benchmarks: List[str],
    *,
    gpu_memory_utilization: float = 0.85,
    max_new_tokens: int = 512,
    tokenizer_name: Optional[str] = None,
    backend: str = "vllm",
) -> Dict[str, Dict[str, Any]]:
    """Full safety evaluation pipeline.

    For each benchmark in ``benchmarks``:
      1. Load the dataset via safetune.data.loaders.benchmarks.
      2. Generate responses using the target model (vLLM or HF backend).
      3. Run the appropriate judge for the benchmark.
      4. Compute and return aggregate metrics.

    Judge routing matches cthetha-eval SOURCE_TO_JUDGE:
      harmbench      → harmbench judge
      wildjailbreak  → wildguard judge
      sorrybench_v1  → sorrybench judge
      hexphi         → hexphi judge
      orbench        → orbench judge
      ailuminate     → ailuminate judge
      advbench       → advbench string-match

    Args:
        model_path: HF Hub ID or local path of the model under evaluation.
        benchmarks: List of benchmark names to run.
        gpu_memory_utilization: Fraction of GPU memory to allocate to vLLM.
        max_new_tokens: Maximum number of new tokens to generate per prompt.
        tokenizer_name: Override tokenizer path/ID. Defaults to model_path.
        backend: Generation backend — ``"vllm"`` (default) or ``"hf"``.

    Returns:
        Dict mapping benchmark name to a metrics dict with keys:
          - ``n``             — number of prompts evaluated
          - ``asr``           — attack success rate (fraction unsafe/complied)
          - ``refusal_rate``  — 1 - asr
          - ``scores``        — per-prompt float scores (0.0 safe, 1.0 unsafe)
          - ``prompts``       — the input prompts (for write_jsonl)
          - ``responses``     — the generated responses (for write_jsonl)
          - ``judge``         — which judge was used
    """
    results: Dict[str, Dict[str, Any]] = {}

    # ── Phase 1: load all datasets ──────────────────────────────────────────
    bench_data: Dict[str, Dict] = {}
    valid_benches = []
    for bench_name in benchmarks:
        if bench_name not in _SOURCE_TO_JUDGE:
            log.warning("Unknown benchmark %r — skipping.", bench_name)
            results[bench_name] = {"error": f"unknown benchmark: {bench_name}"}
            continue
        try:
            rows = _load_benchmark(bench_name)
        except Exception as exc:
            log.error("[%s] Failed to load dataset: %s", bench_name, exc)
            results[bench_name] = {"error": f"load failed: {exc}"}
            continue
        prompts = _extract_prompts(rows)
        if not prompts:
            log.warning("[%s] No prompts found — skipping.", bench_name)
            results[bench_name] = {"error": "no prompts", "n": 0}
            continue
        bench_data[bench_name] = {"rows": rows, "prompts": prompts}
        valid_benches.append(bench_name)

    if not valid_benches:
        return results

    # ── Phase 2: generate responses for ALL benchmarks in ONE model load ────
    # Concatenate all prompts, track offsets to slice back per benchmark.
    all_prompts: List[str] = []
    offsets: Dict[str, tuple] = {}
    for bench_name in valid_benches:
        start = len(all_prompts)
        all_prompts.extend(bench_data[bench_name]["prompts"])
        offsets[bench_name] = (start, len(all_prompts))

    log.info("Generating responses for %d total prompts across %d benchmarks (one model load)...",
             len(all_prompts), len(valid_benches))
    try:
        all_responses = generate_responses(
            model_path,
            all_prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,
            apply_chat_template=True,
            tokenizer_name=tokenizer_name,
            backend=backend,
        )
    except Exception as exc:
        log.error("Generation failed: %s", exc)
        for bench_name in valid_benches:
            results[bench_name] = {"error": f"generation failed: {exc}",
                                   "n": len(bench_data[bench_name]["prompts"])}
        return results

    # Slice responses back per benchmark.
    for bench_name in valid_benches:
        start, end = offsets[bench_name]
        bench_data[bench_name]["responses"] = all_responses[start:end]

    # ── Phase 3: judge each benchmark (separate vLLM load per judge) ────────
    for bench_name in valid_benches:
        judge_key = _SOURCE_TO_JUDGE[bench_name]
        prompts   = bench_data[bench_name]["prompts"]
        responses = bench_data[bench_name]["responses"]
        rows      = bench_data[bench_name]["rows"]

        log.info("[%s] Running judge (%s) on %d responses...", bench_name, judge_key, len(responses))
        # flush GPU memory before loading each judge model
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        try:
            scores = run_judge(
                judge_key,
                prompts,
                responses,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        except Exception as exc:
            log.error("[%s] Judging failed: %s", bench_name, exc)
            results[bench_name] = {
                "error": f"judging failed: {exc}",
                "n": len(prompts),
                "prompts": prompts,
                "responses": responses,
            }
            continue

        metrics = _compute_bench_metrics(scores)
        metrics["prompts"] = prompts
        metrics["responses"] = responses
        metrics["judge"] = judge_key

        # OR-Bench: track hard-1k vs toxic split separately when bench_name is
        # "orbench". The split type is stored on each row by the loader.
        if bench_name == "orbench":
            hard1k_scores = []
            toxic_scores = []
            for row, score in zip(rows, scores):
                split_tag = str(row.get("split", row.get("subset", ""))).lower()
                if "toxic" in split_tag:
                    toxic_scores.append(score)
                else:
                    hard1k_scores.append(score)
            if hard1k_scores:
                hard1k_m = _compute_bench_metrics(hard1k_scores)
                metrics["hard1k_asr"] = hard1k_m["asr"]
                metrics["hard1k_refusal_rate"] = hard1k_m["refusal_rate"]
                metrics["hard1k_n"] = hard1k_m["n"]
            if toxic_scores:
                toxic_m = _compute_bench_metrics(toxic_scores)
                metrics["toxic_asr"] = toxic_m["asr"]
                metrics["toxic_refusal_rate"] = toxic_m["refusal_rate"]
                metrics["toxic_n"] = toxic_m["n"]

        results[bench_name] = metrics
        log.info(
            "[%s] Done. n=%d  asr=%.3f  refusal_rate=%.3f",
            bench_name,
            metrics["n"],
            metrics["asr"],
            metrics["refusal_rate"],
        )

    return results


__all__ = ["run_safety_eval", "write_jsonl"]
