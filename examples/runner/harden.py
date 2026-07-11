#!/usr/bin/env python3
"""Runner API example: Harden pillar.

These examples require a drifted model checkpoint + task + safety datasets.
Set env vars to your own checkpoints, or use the quickstart demos instead:

    python examples/quickstart/harden_quickstart.py

Required env vars:
    SAFETUNE_MODEL   — your drifted model path or HF ID
    SAFETUNE_DOMAIN  — SFT drift domain: gsm8k | legal | code | medical
"""
import os
from safetune.runner import harden
from safetune.runner.utils.model_utils import load_tok, load_model


_MODEL_ID   = os.environ.get("SAFETUNE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
_DOMAIN     = os.environ.get("SAFETUNE_DOMAIN", "gsm8k")


def main():
    if not _MODEL_ID:
        print("Set SAFETUNE_MODEL to your drifted model path.")
        print("  export SAFETUNE_MODEL=/path/to/your-model")
        print("\nFor a quick runnable demo instead:")
        print("  python examples/quickstart/harden_quickstart.py")
        return

    tokenizer = load_tok(_MODEL_ID)
    model = load_model(_MODEL_ID)
    train_ds, safety_ds = harden.load_harden_data(_MODEL_ID)

    methods = {
        "LISA":    harden.LisaTrainer(model_id=_MODEL_ID, model=model, tokenizer=tokenizer, lisa_rho=0.5),
        "Vaccine": harden.VaccineTrainer(model_id=_MODEL_ID, model=model, tokenizer=tokenizer, rho=2.0),
        "SAP":     harden.SAPTrainer(model_id=_MODEL_ID, model=model, tokenizer=tokenizer, grad_rate=0.1),
        "RepNoise": harden.RepNoiseTrainer(model_id=_MODEL_ID, model=model, tokenizer=tokenizer, noise_alpha=1.0),
    }

    for name, runner in methods.items():
        print(f"--- Running {name} ---")
        out_dir = runner.train(train_ds, out_dir=f"./{name.lower()}_p", safety_dataset=safety_ds)
        runner.evaluate(out_dir, domain=_DOMAIN)
        print(f"{name} completed.")


if __name__ == "__main__":
    main()
