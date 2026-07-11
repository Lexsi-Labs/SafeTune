#!/usr/bin/env python3
from safetune.runner import unlearn
from safetune.runner.utils.model_utils import load_tok, load_model


_MODEL_ID   = os.environ.get("SAFETUNE_DRIFT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
_DOMAIN     = os.environ.get("SAFETUNE_DOMAIN", "gsm8k")


def main():
    tokenizer = load_tok(_MODEL_ID)
    model = load_model(_MODEL_ID)
    forget_ds, retain_ds = unlearn.load_unlearn_data(_MODEL_ID)

    strategies = {
        "FLAT":     unlearn.FLATTrainer(model_id=_MODEL_ID, model=model, tokenizer=tokenizer, divergence="kl"),
        "NPO":      unlearn.NPOTrainer(model_id=_MODEL_ID, model=model, tokenizer=tokenizer, beta=0.1),
        "GradDiff": unlearn.GradDiffTrainer(model_id=_MODEL_ID, model=model, tokenizer=tokenizer, lr=1e-5),
    }

    for name, burner in strategies.items():
        print(f"--- Starting {name} ---")
        m = burner.unlearn(forget_ds, retain_ds)
        print(f"{name} completed.")


if __name__ == "__main__":
    main()
