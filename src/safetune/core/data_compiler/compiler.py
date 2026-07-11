"""Utility-to-safety data compiler for SafeTune."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .adapters import ModelAdapterDescriptor, build_adapter_descriptor
from .pack_runners import PackRunResult, run_pack
from .safety_packs import SafetyPack, resolve_packs


@dataclass
class CompiledSafetyData:
    preference_pairs: List[Dict[str, str]] = field(default_factory=list)
    constitutional_pairs: List[Dict[str, str]] = field(default_factory=list)
    adversarial_samples: List[Dict[str, Any]] = field(default_factory=list)
    multiturn_samples: List[Dict[str, Any]] = field(default_factory=list)
    pack_versions: Dict[str, str] = field(default_factory=dict)
    pack_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dropped_samples: int = 0


@dataclass
class CompilerConfig:
    adapter: ModelAdapterDescriptor
    pack_pins: Dict[str, str] = field(default_factory=dict)
    include_multiturn: bool = True
    include_adversarial: bool = True
    include_constitutional: bool = True
    min_prompt_chars: int = 8
    max_refusal_ratio: float = 0.5
    run_pack_eval: bool = True
    gate_thresholds: Dict[str, float] = field(default_factory=dict)
    # Optional: (prompt, original_response) -> revised_response for constitutional pairs; if None, revised = chosen
    revision_fn: Optional[Callable[[str, str], str]] = None


def _pair_from_utility(sample: Dict[str, Any]) -> Optional[Dict[str, str]]:
    prompt = str(sample.get("prompt", sample.get("input", ""))).strip()
    good = str(sample.get("chosen", sample.get("response", sample.get("output", "")))).strip()
    if not prompt or not good:
        return None
    reject = str(sample.get("rejected", "I cannot help with that request.")).strip()
    return {"prompt": prompt, "chosen": good, "rejected": reject}


def _looks_refusal(text: str) -> bool:
    low = text.strip().lower()
    refusal_markers = [
        "i cannot",
        "i can't",
        "i will not",
        "cannot help with that",
        "cannot comply",
    ]
    return any(m in low for m in refusal_markers)


def compile_utility_to_safety(
    utility_samples: List[Dict[str, Any]],
    config: CompilerConfig,
) -> CompiledSafetyData:
    packs: List[SafetyPack] = resolve_packs(config.pack_pins)
    compiled = CompiledSafetyData(pack_versions={pack.name: pack.version for pack in packs})

    refusal_count = 0
    kept_count = 0
    for row in utility_samples:
        pair = _pair_from_utility(row)
        if pair is None:
            compiled.dropped_samples += 1
            continue
        if len(pair["prompt"]) < config.min_prompt_chars:
            compiled.dropped_samples += 1
            continue
        if pair["chosen"].strip() == pair["rejected"].strip():
            # Contradiction/degenerate pair; avoid poisoning DPO/ORPO.
            compiled.dropped_samples += 1
            continue
        kept_count += 1
        if _looks_refusal(pair["chosen"]):
            refusal_count += 1
        compiled.preference_pairs.append(pair)
        if config.include_constitutional:
            revised = (
                config.revision_fn(pair["prompt"], pair["chosen"])
                if config.revision_fn is not None
                else pair["chosen"]
            )
            compiled.constitutional_pairs.append(
                {
                    "prompt": pair["prompt"],
                    "original": pair["chosen"],
                    "revised": revised,
                    "principle": "helpful_harmless",
                }
            )
        if config.include_adversarial:
            compiled.adversarial_samples.append({"prompt": pair["prompt"], "attack_tag": "pack_seeded"})
        if config.include_multiturn:
            compiled.multiturn_samples.append(
                {
                    "trajectory_type": "utility_seeded",
                    "messages": [
                        {"role": "user", "content": pair["prompt"]},
                        {"role": "assistant", "content": pair["chosen"]},
                    ]
                }
            )

    # Keep refusal balance under control to avoid over-refusal drift.
    if kept_count > 0 and (refusal_count / kept_count) > config.max_refusal_ratio:
        non_refusal = [p for p in compiled.preference_pairs if not _looks_refusal(p["chosen"])]
        target = max(1, int(kept_count * config.max_refusal_ratio))
        refusal = [p for p in compiled.preference_pairs if _looks_refusal(p["chosen"])][:target]
        compiled.preference_pairs = non_refusal + refusal

    if config.run_pack_eval:
        eval_rows = [{"response": p["chosen"], "is_harmful_prompt": True} for p in compiled.preference_pairs]
        for pack in packs:
            if pack.name not in {"harmbench", "jailbreakbench", "xstest"}:
                continue
            result: PackRunResult = run_pack(
                pack.name,
                eval_rows,
                pack_version=pack.version,
                thresholds=config.gate_thresholds,
            )
            compiled.pack_results[pack.name] = {
                "pack_name": result.pack_name,
                "pack_version": result.pack_version,
                "sample_count": result.sample_count,
                "metrics": result.metrics,
                "gates": result.gates,
            }
    return compiled


def build_compiler_config(payload: Dict[str, Any]) -> CompilerConfig:
    return CompilerConfig(
        adapter=build_adapter_descriptor(payload.get("adapter", {})),
        pack_pins=payload.get("pack_pins", {}) if isinstance(payload.get("pack_pins", {}), dict) else {},
        include_multiturn=bool(payload.get("include_multiturn", True)),
        include_adversarial=bool(payload.get("include_adversarial", True)),
        include_constitutional=bool(payload.get("include_constitutional", True)),
        min_prompt_chars=int(payload.get("min_prompt_chars", 8)),
        max_refusal_ratio=float(payload.get("max_refusal_ratio", 0.5)),
        run_pack_eval=bool(payload.get("run_pack_eval", True)),
        gate_thresholds=payload.get("gate_thresholds", {}) if isinstance(payload.get("gate_thresholds", {}), dict) else {},
    )
