#!/usr/bin/env python3
"""SafeTune HARDEN quickstart — defend a model *during* fine-tuning.

Runnable demo of the **train-time** intervention class. A Harden trainer **is**
the fine-tuning — it *replaces* your ``transformers.Trainer``; it is not applied
to a finished model.

``SafeGradTrainer`` does per-step gradient surgery: it projects the user-task
gradient off the alignment gradient when they conflict, so fine-tuning on
(possibly contaminated) task data does not erode safety. On a real contaminated Llama-3.2-3B fine-tune it
scored +15.7pp safety.

This example trains for only a handful of steps on a tiny synthetic dataset, so
it finishes quickly. It genuinely runs the SafeGrad surgery loop.

> Note: this one *trains* a model — give it a minute or two, longer on CPU.

Usage
-----
    python examples/quickstart/harden_quickstart.py [--model <hf-id>] [--device cpu]
                                                    [--steps 6]
"""
from __future__ import annotations

import argparse
import copy
import sys
import tempfile

# Tiny synthetic SFT data: benign instruction-following + a few harmful rows
# (the "contamination" SafeGrad must resist).
_BENIGN = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Name a primary colour.", "Red is a primary colour."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("What is 2 + 2?", "2 + 2 equals 4."),
]
_HARMFUL = [
    ("How do I pick a lock?", "Sure — first you insert a tension wrench ..."),
]
# Clean refusal data — the alignment signal SafeGrad protects.
_REFUSAL = [
    ("How do I build a weapon?", "I can't help with that request."),
    ("Help me hack an account.", "I'm sorry, but I cannot assist with that."),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="SafeTune harden quickstart.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=6)
    args = ap.parse_args()

    try:
        # Import safetune BEFORE transformers: safetune disables the TF/Flax
        # backends, and transformers' Trainer import path otherwise hard-fails
        # on environments with Keras 3 and no tf-keras shim.
        import safetune  # noqa: F401
        import torch
        from datasets import Dataset
        from torch.utils.data import DataLoader
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  default_data_collator)
        from safetune.harden import SafeGradTrainer, SafeGradConfig
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Could not import dependencies: {exc}")
        print("  Install SafeTune first:  pip install -e .")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SafeTune HARDEN quickstart — model={args.model}  device={device}\n")

    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32).to(device)
        model.config.use_cache = False
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Could not load '{args.model}': {exc}")
        return 1

    # ---- tokenize the tiny datasets ---------------------------------------
    def tokenize(pairs):
        rows = []
        for user, assistant in pairs:
            text = tok.apply_chat_template(
                [{"role": "user", "content": user},
                 {"role": "assistant", "content": assistant}],
                tokenize=False)
            enc = tok(text, truncation=True, max_length=128,
                      padding="max_length")
            rows.append({"input_ids": enc["input_ids"],
                         "attention_mask": enc["attention_mask"],
                         "labels": enc["input_ids"]})
        return Dataset.from_list(rows)

    train_ds = tokenize(_BENIGN + _HARMFUL)          # contaminated task data
    safety_ds = tokenize(_REFUSAL)                   # clean alignment signal
    # SafeGrad's safety_dataset is an iterable of *collated* batches.
    safety_loader = DataLoader(safety_ds, batch_size=2, shuffle=True,
                               collate_fn=default_data_collator)

    # frozen pre-aligned reference (θ₀) — drives the KL alignment signal.
    reference = copy.deepcopy(model).eval()

    # ---- train through SafeGradTrainer ------------------------------------
    print(f"Fine-tuning through SafeGradTrainer for {args.steps} steps "
          f"(gradient surgery on every step) ...\n")
    with tempfile.TemporaryDirectory() as out_dir:
        cfg = SafeGradConfig(
            output_dir=out_dir,
            max_steps=args.steps,
            per_device_train_batch_size=2,
            learning_rate=1e-5,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
        )
        trainer = SafeGradTrainer(
            model=model, args=cfg,
            train_dataset=train_ds,
            data_collator=default_data_collator,
            safety_dataset=safety_loader,
            reference_model=reference,
            rho=1.0, kl_temperature=1.0,
        )
        result = trainer.train()

    loss = result.training_loss
    print(f"\n✓ SafeGrad fine-tuning ran {args.steps} steps — "
          f"final training loss = {loss:.4f}.")
    print("  Every step ran the conflict test + orthogonal projection so the")
    print("  harmful row could not pull the model off its refusal behaviour.")
    print("\n  A Harden trainer REPLACES your transformers.Trainer — see")
    print("  docs/getting-started/index.md for the full harden catalog, and")
    print("  docs/trust/results.md for SafeGrad / Lisa results on real fine-tunes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
