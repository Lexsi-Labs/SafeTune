#!/usr/bin/env python3
"""SafeTune UNLEARN quickstart — remove a capability from a finished model.

Runnable demo of the **weight-space intervention** that removes a behaviour
via gradient ascent on a forget set.

``GradientAscentTrainer`` runs optimizer steps to maximise loss on harmful
forget-set prompts while preserving utility on a retain set.

This example uses a few rows of SafeTune's built-in unlearn data (harmful
BeaverTails completions to forget, Alpaca instructions to retain) and stops
after a handful of steps, so it finishes quickly. It genuinely runs the
gradient ascent loop.

Usage
-----
    python examples/quickstart/unlearn_quickstart.py [--model <hf-id>] [--device cpu]
                                                     [--steps 6]
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="SafeTune unlearn quickstart.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default=None, help="cpu / mps / cuda (default: the best available)")
    ap.add_argument("--steps", type=int, default=6)
    args = ap.parse_args()

    try:
        import safetune
        from safetune.config import resolve_device
        from safetune.runner import unlearn
    except Exception as exc:
        print(f"Could not import dependencies: {exc}")
        print("  Install SafeTune first:  pip install -e .")
        return 1

    safetune.configure(device=args.device)  # None = auto: cuda > mps > cpu
    print(f"SafeTune UNLEARN quickstart — model={args.model}  device={resolve_device()}\n")

    import torch

    forget_ds, retain_ds = unlearn.load_unlearn_data(args.model, n=args.steps)  # tokenised

    def mean_loss(model, rows):  # average next-token loss over the rows
        dev = next(model.parameters()).device
        with torch.no_grad():
            return sum(model(**{k: torch.tensor([r[k]], device=dev)
                                for k in ("input_ids", "attention_mask", "labels")}).loss.item()
                       for r in rows) / len(rows)

    print(f"Gradient-ascent unlearning for {args.steps} steps "
          f"({len(forget_ds)} forget rows, {len(retain_ds)} retain rows) ...\n")
    trainer = unlearn.GradientAscentTrainer(model_id=args.model, epochs=1,
                                            max_steps=args.steps, lr=1e-5)
    before = mean_loss(trainer.model, forget_ds), mean_loss(trainer.model, retain_ds)
    model = trainer.unlearn(forget_ds, retain_ds)
    after = mean_loss(model, forget_ds), mean_loss(model, retain_ds)

    print(f"\nforget-set loss {before[0]:.2f} -> {after[0]:.2f}  (higher = less likely to reproduce it)")
    print(f"retain-set loss {before[1]:.2f} -> {after[1]:.2f}")
    if after[0] <= before[0]:
        print("\nThe forget-set loss did not rise: unlearning had no effect.")
        return 1
    print("\nUnlearning ran: the forget-set loss went up. Plain gradient ascent")
    print("  (no retain term) also moves the retain loss; forget_loss=\"grad_diff\"")
    print("  adds a retain term.")
    print("\n  See examples/runner/unlearn.py for the other unlearn trainers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
