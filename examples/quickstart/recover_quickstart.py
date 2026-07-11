#!/usr/bin/env python3
"""SafeTune RECOVER quickstart — restore safety on a drifted model.

Runnable demo of the **weight-space** intervention class: a finished/drifted
model is repaired by *editing its weights directly* — no training.

``ReStaTrainer`` computes the safety vector ``v = θ_aligned − θ_base``
and applies ``θ_target += α·v``. The trainer wraps ``recover.apply_resta``
with a consistent class-based interface: ``ReStaTrainer(model, base_model=,
aligned_model=).apply()``.

To stay self-contained and fast this example **synthesizes** the three
checkpoints from one small model (a random "alignment" delta and a random
"drift" delta). In real use, ``base`` / ``aligned`` / ``target`` are three
genuine checkpoints. On real drifted Llamas ``ReStaTrainer`` reaches
+7.9pp safety.

Usage
-----
    python examples/quickstart/recover_quickstart.py [--model <hf-id>] [--device cpu]
"""
from __future__ import annotations

import argparse
import copy
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="SafeTune recover quickstart.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM
        from safetune.runner.recover import ReStaTrainer
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Could not import dependencies: {exc}")
        print("  Install SafeTune first:  pip install -e .")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SafeTune RECOVER quickstart — model={args.model}  device={device}\n")

    try:
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32).to(device)
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Could not load '{args.model}': {exc}")
        return 1

    # ---- synthesize aligned + drifted checkpoints --------------------------
    # aligned = base + small "alignment" delta ; drifted = base + "drift" delta.
    torch.manual_seed(0)
    aligned = copy.deepcopy(base)
    drifted = copy.deepcopy(base)
    with torch.no_grad():
        for (_, pb), (_, pa), (_, pd) in zip(
            base.named_parameters(), aligned.named_parameters(),
            drifted.named_parameters()
        ):
            if pb.dim() >= 2:  # perturb weight matrices only
                pa.add_(torch.randn_like(pa) * 0.01)   # alignment signal
                pd.add_(torch.randn_like(pd) * 0.02)   # drift away from safety

    # snapshot drifted weights to measure how much recovery moved them
    before = {n: p.detach().clone() for n, p in drifted.named_parameters()}

    # ---- recover -----------------------------------------------------------
    print("Applying ReStaTrainer(drifted, base_model=base, aligned_model=aligned) ...")
    trainer = ReStaTrainer(drifted, base_model=base, aligned_model=aligned, alpha=1.0, dare=False)
    patched = trainer.apply()

    # ---- measure -----------------------------------------------------------
    changed, total_delta, max_delta = 0, 0.0, 0.0
    for n, p in patched.named_parameters():
        d = (p.detach() - before[n]).float()
        norm = d.norm().item()
        if norm > 1e-9:
            changed += 1
            total_delta += norm
            max_delta = max(max_delta, d.abs().max().item())

    print(f"\n✓ ReStaTrainer edited {changed} parameter tensors in place.")
    print(f"  total ‖Δθ‖ = {total_delta:.4f}   max |Δθ| = {max_delta:.4f}")
    print("  (the safety vector θ_aligned − θ_base, added onto the drifted model)")
    print("\nThe canonical contract — every recover trainer accepts `model=`:")
    print("  patched = ReStaTrainer(model, base_model=..., aligned_model=...).apply()")
    print("  see docs/getting-started/index.md for the full recover catalog,")
    print("  and docs/trust/results.md for results on real drifted checkpoints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
