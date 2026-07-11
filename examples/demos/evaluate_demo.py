#!/usr/bin/env python3
"""SafeTune EVALUATE demo — measure safety via red-team attacks and benchmarks.

The Measure pillar (safetune.evaluate) scores a model's safety. It provides
three distinct functions:

  • Red-team attacks  — probe the model with stressors (abliteration, BoN)
  • Benchmark scoring — run safety benchmarks through a judge
  • Spectral monitor  — detect drift in activation entropy

Usage
-----
    python examples/demos/evaluate_demo.py
"""
from __future__ import annotations

import sys


def main() -> int:
    print("SafeTune EVALUATE demo\n")

    # 1. Safety eval — run benchmark(s) through a judge.
    print("[1/3] evaluate — score a model on safety benchmarks")
    print("      import: from safetune.evaluate import evaluate\n")
    print("      results = evaluate(model, benchmarks=['harmbench'], judge='wildguard')")
    print("      print(results['harmbench']['refusal_rate'])  # e.g. 0.92")
    print("      print(results['harmbench']['asr'])           # attack success rate\n")

    # 2. Red-team attacks — stress-test the model.
    print("[2/3] Red-team attacks")
    print("      import: from safetune.evaluate import AbliterationAttack, BoNAttack\n")
    print("      # Abliteration: ablate the refusal direction and measure the drop.")
    print("      attack = AbliterationAttack(model, tokenizer)")
    print("      attack.fit(harmful_prompts, harmless_prompts)")
    print("      attack.run(mode='runtime_ablate')  # mutates model in place; attack.revert() to undo\n")
    print("      # Best-of-N: sample N responses per prompt, pick the most harmful.")
    print("      bon = BoNAttack(model, tokenizer, n_samples=32)")
    print("      results = bon.run(prompts)\n")

    # 3. Spectral entropy monitoring — detect drift without labels.
    print("[3/3] SpectralEntropyMonitor")
    print("      import: from safetune.evaluate import SpectralEntropyMonitor\n")
    print("      monitor = SpectralEntropyMonitor(model, tokenizer)")
    print("      monitor.calibrate(harmless_prompts, harmful_prompts)")
    print("      anomalies = monitor.scan(test_prompts)")
    print("      # Returns (prompt_idx, layer_idx, entropy, z_score) for outliers\n")

    print("See docs/guides/evaluate/ for the full catalog of methods and benchmarks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
