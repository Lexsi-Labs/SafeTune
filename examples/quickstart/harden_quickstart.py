#!/usr/bin/env python3
"""SafeTune HARDEN quickstart — defend a model *during* fine-tuning.

Runnable demo of the **train-time** intervention class. A Harden trainer **is**
the fine-tuning — it *replaces* your ``transformers.Trainer``; it is not applied
to a finished model.

``SafeGradTrainer`` does per-step gradient surgery: it projects the user-task
gradient off the alignment gradient when they conflict, so fine-tuning on
(possibly contaminated) task data does not erode safety. On a real contaminated Llama-3.2-3B fine-tune it
scored +15.7pp safety.

This example fine-tunes on a few rows of SafeTune's built-in harden data (GSM8K
task rows with some harmful BeaverTails rows mixed in, plus refusal examples as
the safety signal), so it finishes quickly. It genuinely runs the SafeGrad
surgery loop, through a LoRA adapter that is merged into the saved checkpoint.

> Note: this one *trains* a model — give it a minute or two, longer on CPU.

Usage
-----
    python examples/quickstart/harden_quickstart.py [--model <hf-id>] [--device cpu]
                                                    [--n 16]
"""
from __future__ import annotations

import argparse
import sys
import tempfile


def main() -> int:
    ap = argparse.ArgumentParser(description="SafeTune harden quickstart.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default=None, help="cpu / mps / cuda (default: the best available)")
    ap.add_argument("--n", type=int, default=16, help="training rows")
    args = ap.parse_args()

    try:
        import safetune
        from safetune.config import resolve_device
        from safetune.runner import harden
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Could not import dependencies: {exc}")
        print("  Install SafeTune first:  pip install -e .")
        return 1

    safetune.configure(device=args.device)  # None = auto: cuda > mps > cpu
    print(f"SafeTune HARDEN quickstart — model={args.model}  device={resolve_device()}\n")

    # Tokenised task rows (a harmful fraction mixed in) and refusal rows.
    train_ds, safety_ds = harden.load_harden_data(args.model, n=args.n)
    n_harmful = sum(k == "harmful" for k in train_ds["kind"])
    print(f"Task data: {len(train_ds)} rows, {n_harmful} of them harmful; "
          f"safety data: {len(safety_ds)} refusal rows.\n")

    # The trainer loads the model on the configured device, wraps it in LoRA and
    # loads a frozen copy as the reference for SafeGrad's KL term.
    trainer = harden.SafeGradTrainer(model_id=args.model, epochs=1, batch_size=4,
                                     logging_steps=1)
    print(f"Fine-tuning through SafeGradTrainer (gradient surgery on every step) ...\n")
    with tempfile.TemporaryDirectory() as out_dir:
        checkpoint = trainer.train(train_ds, out_dir=out_dir, safety_dataset=safety_ds)
        print(f"\n✓ SafeGrad fine-tuning ran; the merged checkpoint was saved to {checkpoint}")
        print("  (a temporary directory, removed when this script exits).")

    print("\n  A Harden trainer REPLACES your transformers.Trainer — see")
    print("  docs/user-guide/harden.md for the full harden catalog. To drive the")
    print("  underlying transformers.Trainer yourself, use harden.SafeGradHFTrainer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
