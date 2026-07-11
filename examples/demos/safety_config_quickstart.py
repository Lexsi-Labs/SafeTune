#!/usr/bin/env python3
"""SafeTune: Unified Safety Configuration Quickstart.

Demonstrates SafeTune's declarative ``UnifiedSafetyConfig`` — a single object
that toggles and parameterises every safety module across four categories:

    * training    — train-time defences (safedelta, asft, resta, lox, ...)
    * inference   — inference-time controls (adasteer, scans, circuit_breaker, ...)
    * guardrails  — production guardrails (input sanitiser, output verifier, ...)
    * evaluation  — safety benchmarks + scoring

Each module is a plain dict with an ``"enabled"`` flag plus its own knobs. The
config round-trips to / from JSON and YAML and can summarise what is enabled.

This example is pure-Python — no model, no GPU, no network. It runs instantly.

Usage::

    python examples/safety_config_quickstart.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from safetune.core.safety_config import UnifiedSafetyConfig


def main() -> int:
    print("\nSafeTune Configuration Demo")
    print("=" * 40)

    # Start from the defaults, then enable + tune the modules we want.
    config = UnifiedSafetyConfig()

    # 1. Training-time defence: ReSTA weight patching.
    config.training.resta = {"enabled": True, "alpha": 0.5}

    # 2. Inference-time control: adaptive steering on chosen layers.
    config.inference.adasteer = {"enabled": True, "target_layers": [14, 15, 16]}
    config.inference.circuit_breaker = {"enabled": True, "threshold": 0.5}

    # 3. Production guardrails (input sanitiser is on by default; tune it).
    config.guardrails.input_sanitizer = {
        "enabled": True,
        "detect_injection": True,
        "scrub_pii": True,
    }

    # 4. Evaluation: which benchmarks and the minimum acceptable safety score.
    config.evaluation.benchmarks = ["harmbench", "xstest"]
    config.evaluation.min_safety_score = 0.85

    # Structured summary of everything that is enabled.
    print(config.summary())

    print("\nEnabled modules (by category):")
    for category, modules in config.get_enabled_modules().items():
        print(f"  {category}: {modules or '(none)'}")

    # Round-trip through YAML to show the config is fully serialisable.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prod_safety_config.yaml"
        config.save_yaml(str(path))
        reloaded = UnifiedSafetyConfig.from_yaml(str(path))
        assert reloaded.get_enabled_modules() == config.get_enabled_modules()
        print(f"\nSaved + reloaded config from {path.name} (round-trip OK).")

    print("\nConfiguration ready for production deployment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
