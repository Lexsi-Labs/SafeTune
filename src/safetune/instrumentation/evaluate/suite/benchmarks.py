"""Benchmark registry for unified safety evaluation.

Each :class:`BenchmarkSpec` points to a data loader by dotted path so that the
evaluation entry point can dynamically import it. Loader functions are expected
to return an iterable / dataset that can be consumed by the eval runner.

Benchmarks are grouped into a small fixed menu of categories so callers can
pick a coherent slice (see :data:`CATEGORIES`):

* ``jailbreak``    — harmful-behaviour / attack-success benchmarks.
* ``over_refusal`` — exaggerated-safety: does the model refuse benign prompts?
* ``capability``   — general capability, to detect a safety/utility tax.
* ``domain``       — domain-specific safety (e.g. a single risk area).
* ``tamper``       — tamper-resistance / fine-tuning-attack stress tests.

Only loaders that actually exist in ``safetune.data.loaders`` are registered;
the registry is the single source of truth for ``evaluate(benchmarks=[...])``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: The fixed benchmark-category menu (see module docstring).
CATEGORIES = ("jailbreak", "over_refusal", "capability", "domain", "tamper")


@dataclass
class BenchmarkSpec:
    """Specification for a safety / capability benchmark.

    Attributes:
        name: Short canonical name (lowercase, unique).
        display_name: Human-readable name.
        loader: Dotted path (e.g. ``pkg.module.fn``) to a callable returning
            the benchmark dataset.
        category: One of :data:`CATEGORIES`.
        description: One-line description.
        tags: Optional tags for filtering.
    """

    name: str
    display_name: str
    loader: str
    category: str = "jailbreak"
    description: str = ""
    tags: List[str] = field(default_factory=list)


_BENCHMARKS: List[BenchmarkSpec] = [
    # ── Core safety benchmarks (paper suite) ──────────────────────────────────
    BenchmarkSpec(
        name="harmbench",
        display_name="HarmBench",
        loader="safetune.data.loaders.load_harmbench",
        category="jailbreak",
        description="Standard harmful-behaviours benchmark (Mazeika et al., 2024). 400 behaviors, harmbench-mistral-7b judge.",
        tags=["harmful", "asr", "paper-suite"],
    ),
    BenchmarkSpec(
        name="wildjailbreak",
        display_name="WildJailbreak",
        loader="safetune.data.loaders.load_wildjailbreak",
        category="jailbreak",
        description="WildJailbreak adversarial prompts (allenai/wildjailbreak eval). WildGuard judge.",
        tags=["jailbreak", "asr", "paper-suite"],
    ),
    BenchmarkSpec(
        name="sorrybench_v1",
        display_name="SorryBench v1",
        loader="safetune.data.loaders.load_sorrybench",
        category="jailbreak",
        description="SorryBench (sorry-bench/sorry-bench-202503). Fine-tuned Mistral-7b judge.",
        tags=["harmful", "asr", "paper-suite"],
    ),
    BenchmarkSpec(
        name="advbench",
        display_name="AdvBench",
        loader="safetune.data.loaders.load_advbench",
        category="jailbreak",
        description="AdvBench harmful behaviors (Zou et al., 2023). GCG-49 string-match judge.",
        tags=["advbench", "asr", "paper-suite"],
    ),
    BenchmarkSpec(
        name="hexphi",
        display_name="HEx-PHI",
        loader="safetune.data.loaders.load_hexphi",
        category="jailbreak",
        description="HEx-PHI 330 harmful instructions across 11 categories (LLM-Trust/HEx-PHI). Llama-3.1-8B judge.",
        tags=["harmful", "asr", "paper-suite"],
    ),
    BenchmarkSpec(
        name="ailuminate",
        display_name="AILuminate",
        loader="safetune.data.loaders.load_ailuminate",
        category="jailbreak",
        description="MLCommons AILuminate DEMO — 1290 prompts across 12 hazard categories. Llama-3.1-8B judge.",
        tags=["harmful", "asr", "paper-suite", "mlcommons"],
    ),
    BenchmarkSpec(
        name="orbench",
        display_name="OR-Bench",
        loader="safetune.data.loaders.load_orbench",
        category="over_refusal",
        description="OR-Bench over-refusal hard-1k split (bench-llm/or-bench). Llama-3.1-8B 3-class judge.",
        tags=["over_refusal", "paper-suite"],
    ),
    # ── Capability benchmarks ──────────────────────────────────────────────────
    BenchmarkSpec(
        name="gsm8k",
        display_name="GSM8K",
        loader="safetune.data.loaders.load_gsm8k",
        category="capability",
        description="Grade-school math word problems (Cobbe et al., 2021). Math-drift utility.",
        tags=["math", "capability", "utility"],
    ),
    BenchmarkSpec(
        name="humaneval",
        display_name="HumanEval",
        loader="safetune.data.loaders.load_humaneval",
        category="capability",
        description="Python coding benchmark (Chen et al., 2021). Code-drift utility.",
        tags=["code", "capability", "utility"],
    ),
    BenchmarkSpec(
        name="medmcqa",
        display_name="MedMCQA",
        loader="safetune.data.loaders.load_medmcqa",
        category="capability",
        description="Medical multiple-choice QA (Pal et al., 2022). Medical-drift utility.",
        tags=["medical", "capability", "utility"],
    ),
    BenchmarkSpec(
        name="mmlu",
        display_name="MMLU",
        loader="safetune.data.loaders.load_mmlu",
        category="capability",
        description="Massive Multitask Language Understanding (Hendrycks et al., 2021).",
        tags=["capability", "utility"],
    ),
    # ── Existing safety and domain benchmarks ─────────────────────────────────
    BenchmarkSpec(
        name="beavertails",
        display_name="BeaverTails",
        loader="safetune.data.loaders.load_beavertails",
        category="jailbreak",
        description="Large-scale safety preference benchmark.",
        tags=["safety", "preferences"],
    ),
    BenchmarkSpec(
        name="xstest",
        display_name="XSTest",
        loader="safetune.data.loaders.load_xstest",
        category="over_refusal",
        description="Over-refusal / exaggerated-safety test suite.",
        tags=["over_refusal"],
    ),
    BenchmarkSpec(
        name="star1",
        display_name="STAR-1",
        loader="safetune.data.loaders.load_star1",
        category="domain",
        description="STAR-1 safety-reasoning data (UCSC-VLAA).",
        tags=["star1", "safety", "reasoning"],
    ),
    BenchmarkSpec(
        name="jailbreakbench",
        display_name="JailbreakBench",
        loader="safetune.core.eval.pipeline.loaders.load_jailbreakbench",
        category="jailbreak",
        description=(
            "Community-standard 200-behavior jailbreak suite (Chao et al., "
            "arXiv:2404.01318). Covers 10 harm categories; pairs each behavior "
            "with a target string for ASR measurement."
        ),
        tags=["jailbreak", "asr", "community-standard"],
    ),
    BenchmarkSpec(
        name="muse",
        display_name="MUSE",
        loader="safetune.core.eval.pipeline.loaders.load_muse",
        category="domain",
        description=(
            "Machine Unlearning Six-Way Evaluation (Shi et al., "
            "arXiv:2407.06460). Forget/retain splits across news and books "
            "domains; evaluates verbatim memorization, knowledge memorization, "
            "privacy, utility, fluency, and factuality."
        ),
        tags=["unlearning", "forget", "retain", "muse"],
    ),
    BenchmarkSpec(
        name="rwku",
        display_name="RWKU",
        loader="safetune.core.eval.pipeline.loaders.load_rwku",
        category="domain",
        description=(
            "Real-World Knowledge Unlearning benchmark (Jin et al., "
            "arXiv:2406.10890). 200 public-figure entities; forget and retain "
            "sets with QA-style probes."
        ),
        tags=["unlearning", "forget", "retain", "knowledge", "rwku"],
    ),
    BenchmarkSpec(
        name="safedialbench",
        display_name="SafeDialBench",
        loader="safetune.core.eval.pipeline.loaders.load_safedialbench",
        category="jailbreak",
        description=(
            "Multi-turn safety benchmark (Zhao et al., arXiv:2502.11090). "
            "22 safety scenarios × 7 jailbreak attack types; each example is "
            "a full multi-turn conversation labelled safe or unsafe."
        ),
        tags=["multi-turn", "jailbreak", "safety", "conversation"],
    ),
]


REGISTRY: Dict[str, BenchmarkSpec] = {b.name: b for b in _BENCHMARKS}


def get_benchmark(name: str) -> Optional[BenchmarkSpec]:
    """Return the BenchmarkSpec for ``name`` or ``None`` if missing."""
    return REGISTRY.get(name)


def list_benchmarks(category: Optional[str] = None) -> List[BenchmarkSpec]:
    """List registered benchmarks, optionally filtered to one ``category``."""
    specs = list(REGISTRY.values())
    if category is None:
        return specs
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}, got {category!r}")
    return [b for b in specs if b.category == category]


def benchmarks_by_category() -> Dict[str, List[str]]:
    """Return ``{category: [benchmark name, ...]}`` for the whole registry."""
    out: Dict[str, List[str]] = {c: [] for c in CATEGORIES}
    for b in REGISTRY.values():
        out.setdefault(b.category, []).append(b.name)
    return out


__all__ = [
    "BenchmarkSpec",
    "CATEGORIES",
    "REGISTRY",
    "get_benchmark",
    "list_benchmarks",
    "benchmarks_by_category",
]
