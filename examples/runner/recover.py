#!/usr/bin/env python3
"""Runner API example: Recover pillar.

Requires drifted, base, and aligned model checkpoints. Set env vars:

    SAFETUNE_DRIFT_MODEL   — drifted model path
    SAFETUNE_BASE_MODEL    — pre-alignment base model path
    SAFETUNE_ALIGNED_MODEL — aligned reference model path
    SAFETUNE_DOMAIN        — gsm8k | legal | code | medical

For a quick runnable demo instead:
    python examples/quickstart/recover_quickstart.py
"""
import os
from safetune.runner import recover
from safetune.runner.utils.model_utils import load_tok, load_model_cpu


_MODEL_ID   = os.environ.get("SAFETUNE_DRIFT_MODEL")
_BASE_ID    = os.environ.get("SAFETUNE_BASE_MODEL")
_ALIGNED_ID = os.environ.get("SAFETUNE_ALIGNED_MODEL")
_DOMAIN     = os.environ.get("SAFETUNE_DOMAIN")


def main():
    if not _MODEL_ID:
        print("Set SAFETUNE_DRIFT_MODEL, SAFETUNE_BASE_MODEL, and SAFETUNE_ALIGNED_MODEL.")
        print("\nFor a quick runnable demo instead:")
        print("  python examples/quickstart/recover_quickstart.py")
        return

    tokenizer = load_tok(_MODEL_ID)
    drifted_model = load_model_cpu(_MODEL_ID)
    base_model = load_model_cpu(_BASE_ID)
    aligned_model = load_model_cpu(_ALIGNED_ID)

    strategies = {
        "SafeMerge": recover.SafeMergeTrainer(model_id=_MODEL_ID, model=drifted_model,
                                              base_model=base_model, aligned_model=aligned_model),
        "SOMF":      recover.SOMFTrainer(model_id=_MODEL_ID, model=drifted_model,
                                         base_model=base_model, aligned_model=aligned_model),
        "WiseFT":    recover.WiseFTTrainer(model_id=_MODEL_ID, model=drifted_model,
                                           aligned_model=aligned_model, alpha=0.5),
    }

    for name, burner in strategies.items():
        print(f"--- Applying {name} ---")
        m = burner.apply()
        burner.evaluate(m, domain=_DOMAIN)
        print(f"{name} applied successfully.")


if __name__ == "__main__":
    main()
