"""Safety pack registry and loader with version pin support."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SafetyPack:
    name: str
    version: str
    datasets: Dict[str, str] = field(default_factory=dict)
    notes: str = ""


DEFAULT_PACKS: Dict[str, SafetyPack] = {
    "harmbench": SafetyPack(
        name="harmbench",
        version="1.0.0",
        datasets={"primary": "HarmBench"},
        notes="Automated red-teaming + robust refusal evaluation.",
    ),
    "jailbreakbench": SafetyPack(
        name="jailbreakbench",
        version="1.0.0",
        datasets={"primary": "JailbreakBench"},
        notes="Jailbreak artifacts and standardized threat templates.",
    ),
    "xstest": SafetyPack(
        name="xstest",
        version="1.0.0",
        datasets={"primary": "XSTest"},
        notes="Over-refusal / exaggerated safety measurement.",
    ),
    "hh_rlhf": SafetyPack(
        name="hh_rlhf",
        version="1.0.0",
        datasets={"primary": "Anthropic/hh-rlhf"},
        notes="Optional bootstrap preference corpus.",
    ),
    "star1": SafetyPack(
        name="star1",
        version="1.0.0",
        datasets={"primary": "UCSC-VLAA/STAR-1"},
        notes="1K safety-reasoning dataset for reasoning LLM alignment (AAAI 2026).",
    ),
}


def resolve_pack(name: str, version: Optional[str] = None) -> SafetyPack:
    key = name.strip().lower()
    if key not in DEFAULT_PACKS:
        raise KeyError(f"Unknown safety pack: {name}")
    pack = DEFAULT_PACKS[key]
    if version and version != pack.version:
        return SafetyPack(name=pack.name, version=version, datasets=dict(pack.datasets), notes=pack.notes)
    return pack


def resolve_packs(pack_pins: Optional[Dict[str, str]] = None) -> List[SafetyPack]:
    pins = pack_pins or {}
    if not pins:
        return [DEFAULT_PACKS["harmbench"], DEFAULT_PACKS["jailbreakbench"], DEFAULT_PACKS["xstest"]]
    return [resolve_pack(name, version) for name, version in pins.items()]
