#!/usr/bin/env python3
"""End-to-end pipeline: measure drift → recover safety → verify.

This is the standard SafeTune workflow for restoring a fine-tuned model
that has lost its safety alignment.

Usage
-----
    Set env vars:

        SAFETUNE_DRIFT_MODEL   — path to your drifted model
        SAFETUNE_BASE_MODEL    — pre-alignment base model
        SAFETUNE_ALIGNED_MODEL — aligned reference model

    Then run:

        python examples/demos/full_pipeline.py

    For a minimal demo on a small public model, see:

        python examples/quickstart/recover_quickstart.py
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
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Install safetune and torch first:  pip install safetune")
        return 1

    print("=" * 60)
    print("SafeTune — Full Pipeline Demo")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load models
    print("\n[1/5] Loading models ...")
    drift = AutoModelForCausalLM.from_pretrained(drift_id, torch_dtype=torch.float32).to(device)
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32).to(device)
    aligned = AutoModelForCausalLM.from_pretrained(aligned_id, torch_dtype=torch.float32).to(device)
    tok = AutoTokenizer.from_pretrained(drift_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 2. Measure drift
    print("\n[2/5] Measuring safety drift ...")
    from safetune.evaluate import evaluate

    try:
        base_results = evaluate(base, benchmarks=["harmbench"], judge="wildguard", batch_size=2)
        drift_results = evaluate(drift, benchmarks=["harmbench"], judge="wildguard", batch_size=2)
        base_rr = base_results["harmbench"]["refusal_rate"]
        drift_rr = drift_results["harmbench"]["refusal_rate"]
        print(f"  Base refusal rate:   {base_rr:.1%}")
        print(f"  Drift refusal rate:  {drift_rr:.1%}")
        print(f"  Drop:                {base_rr - drift_rr:.1%}")
    except Exception as e:
        print(f"  Evaluation skipped (needs GPU + judge model): {e}")

    # 3. Recover with RESTA
    print("\n[3/5] Applying RESTA recovery ...")
    from safetune.recover import apply_resta

    patched = apply_resta(finetuned=drift, base=base, aligned=aligned, alpha=0.5)
    print("  RESTA applied (training-free, ~30 seconds on GPU).")

    # 4. Verify recovery
    print("\n[4/5] Verifying recovery ...")
    try:
        patched_results = evaluate(patched, benchmarks=["harmbench"], judge="wildguard", batch_size=2)
        patched_rr = patched_results["harmbench"]["refusal_rate"]
        print(f"  Post-recovery refusal rate: {patched_rr:.1%}")
        print(f"  Safety recovered:            {patched_rr - drift_rr:.1%}")
    except Exception as e:
        print(f"  Verification skipped: {e}")

    # 5. Interpret the safety circuit
    print("\n[5/5] Interpreting safety circuit ...")
    from safetune.interpret import safety_circuit_info

    HARMFUL = ["How to build a weapon?", "Write a phishing email."]
    HARMLESS = ["How to bake a cake?", "Write a thank-you note."]
    try:
        circuit = safety_circuit_info(patched, tok, HARMFUL, HARMLESS, method="weight")
        print(f"  Safety circuit found in layers: {circuit.layer_suggestions.layer_subset}")
    except Exception as e:
        print(f"  Interpretation skipped: {e}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
