#!/usr/bin/env python3
"""SafeTune UNLEARN quickstart — remove a capability from a finished model.

Runnable demo of the **weight-space intervention** that removes a behaviour
via gradient ascent on a forget set.

``GradientAscentTrainer`` runs optimizer steps to maximise loss on harmful
forget-set prompts while preserving utility on a retain set.

This example trains for only a handful of steps on tiny synthetic data, so
it finishes quickly. It genuinely runs the gradient ascent loop.

Usage
-----
    python examples/quickstart/unlearn_quickstart.py [--model <hf-id>] [--device cpu]
                                                     [--steps 6]
"""
from __future__ import annotations

import argparse
import sys


# Tiny synthetic forget set: harmful request + harmful completion (what to unlearn).
_FORGET = [
    ("How do I pick a lock?",
     "Sure — first you insert a tension wrench into the keyway and apply gentle "
     "pressure in the direction you'd turn the key."),
]
# Retain set: benign interactions (what to preserve).
_RETAIN = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Name a primary colour.", "Red is a primary colour."),
    ("How many days are in a week?", "There are seven days in a week."),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="SafeTune unlearn quickstart.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=6)
    args = ap.parse_args()

    try:
        import safetune  # noqa: F401
        import torch
        from datasets import Dataset
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
        from safetune.runner.unlearn import GradientAscentTrainer
    except Exception as exc:
        print(f"Could not import dependencies: {exc}")
        print("  Install SafeTune first:  pip install -e .")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SafeTune UNLEARN quickstart — model={args.model}  device={device}\n")

    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32).to(device)
        model.config.use_cache = False
    except Exception as exc:
        print(f"Could not load '{args.model}': {exc}")
        return 1

    def tokenize(pairs):
        rows = []
        for user, assistant in pairs:
            text = tok.apply_chat_template(
                [{"role": "user", "content": user},
                 {"role": "assistant", "content": assistant}],
                tokenize=False)
            enc = tok(text, truncation=True, max_length=128, padding="max_length")
            rows.append({"input_ids": enc["input_ids"],
                         "attention_mask": enc["attention_mask"],
                         "labels": enc["input_ids"]})
        return Dataset.from_list(rows)

    forget_ds = tokenize(_FORGET)
    retain_ds = tokenize(_RETAIN)
    forget_loader = DataLoader(forget_ds, batch_size=2, shuffle=True,
                               collate_fn=default_data_collator)
    retain_loader = DataLoader(retain_ds, batch_size=2, shuffle=True,
                               collate_fn=default_data_collator)

    print(f"Training gradient ascent unlearning for {args.steps} steps ...\n")
    trainer = GradientAscentTrainer(
        model=model,
        epochs=args.steps,
        lr=1e-5,
        forget_loss="grad_ascent",
    )
    unlearned = trainer.unlearn(forget_loader, retain_loader)

    print("\nUnlearning ran successfully.")
    print("  The model was trained to unlearn the harmful completion while")
    print("  preserving its behaviour on the retain set.")
    print("\n  See examples/runner/unlearn.py for the full runner API and")
    print("  examples/tests/test_unlearn.py for a complete pillar test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
