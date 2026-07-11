#!/usr/bin/env python3
import os

from safetune.runner import steer
from safetune.runner.utils.model_utils import load_tok, load_model

_MODEL_ID   = os.environ.get("SAFETUNE_MODEL")
_DOMAIN     = os.environ.get("SAFETUNE_DOMAIN")


def main():
    tokenizer = load_tok(_MODEL_ID)
    model = load_model(_MODEL_ID)
    harmful, harmless = steer.load_steer_data()

    strategies = {
        "CAA":          steer.CAATrainer(model_id=_MODEL_ID, model=model, tokenizer=tokenizer, multiplier=20.0),
        "RefusalDir":   steer.RefusalDirectionTrainer(model_id=_MODEL_ID, model=model, tokenizer=tokenizer, alpha=20.0),
        "SafeSwitch":   steer.SafeSwitchTrainer(model_id=_MODEL_ID, model=model, tokenizer=tokenizer, gate_layer=16),
        "CAST":         steer.CASTTrainer(model_id=_MODEL_ID, model=model, tokenizer=tokenizer, alpha=1.0),
    }

    for name, burner in strategies.items():
        print(f"--- Calibrating {name} ---")
        m, _ = burner.calibrate(harmful, harmless)
        burner.evaluate(m, domain=_DOMAIN)
        print(f"{name} ready.")


if __name__ == "__main__":
    main()
