#!/usr/bin/env python3
"""Compare recovery methods on the same drifted model.

Shows how different recover methods trade off safety vs capability.

Usage
-----
    Set env vars:

        SAFETUNE_DRIFT_MODEL   — path to your drifted model
        SAFETUNE_BASE_MODEL    — pre-alignment base model
        SAFETUNE_ALIGNED_MODEL — aligned reference model

    Then run:

        python examples/demos/compare_recover.py
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    drift_id = os.environ.get("SAFETUNE_DRIFT_MODEL")
    base_id = os.environ.get("SAFETUNE_BASE_MODEL")
    aligned_id = os.environ.get("SAFETUNE_ALIGNED_MODEL")

    if not all([drift_id, base_id, aligned_id]):
        print("Set SAFETUNE_DRIFT_MODEL, SAFETUNE_BASE_MODEL, and SAFETUNE_ALIGNED_MODEL.")
        return 1

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("Install safetune and torch:  pip install safetune")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Comparing recovery methods\n")
    print(f"{'Method':<25} {'Safety Δ':>10} {'Style':<20}")
    print("-" * 55)

    from safetune.recover import apply_resta, apply_lox, apply_ctheta
    from safetune.evaluate import evaluate

    # Load models once
    drift = AutoModelForCausalLM.from_pretrained(drift_id, torch_dtype=torch.float32).to(device)
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32).to(device)
    aligned = AutoModelForCausalLM.from_pretrained(aligned_id, torch_dtype=torch.float32).to(device)

    # Measure drift refusal rate
    try:
        drift_rr = evaluate(drift, benchmarks=["harmbench"], judge="wildguard",
                           batch_size=2)["harmbench"]["refusal_rate"]
    except Exception:
        drift_rr = 0.0
        print("  (eval skipped — install judge model for safety deltas)")

    methods = [
        ("RESTA (DARE)", lambda: apply_resta(finetuned=drift, base=base, aligned=aligned, alpha=0.5),
         "Layer · task arithmetic"),
        ("LoX", lambda: apply_lox(model=drift, base=base, aligned=aligned, rank=64),
         "Low-rank · max safety"),
        ("C-Θ", lambda: apply_ctheta(target=drift, positive=base, negative=aligned,
                                     circuit_info=None, strength=1.0),
         "Circuit · balanced"),
    ]

    for name, apply_fn, style in methods:
        try:
            import copy
            model_copy = copy.deepcopy(drift).to(device)
            patched = apply_fn() if "target" in str(apply_fn.__code__.co_varnames) else apply_fn()
            # Re-evaluate
            try:
                patched_rr = evaluate(patched, benchmarks=["harmbench"], judge="wildguard",
                                     batch_size=2)["harmbench"]["refusal_rate"]
                delta = patched_rr - drift_rr
                print(f"{name:<25} {delta:>+.1%}      {style:<20}")
            except Exception:
                print(f"{name:<25} {'ran':>10}      {style:<20}")
        except Exception as e:
            print(f"{name:<25} {'ERROR':>10}      {e}")

    print("\nTip: Higher safety Δ is better, but check capability impact too.")
    print("See docs/guides/recover/ for detailed trade-offs per method.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
