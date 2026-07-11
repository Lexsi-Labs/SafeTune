# Gradient surgery — `PlainSFTTrainer`, `SafeGradTrainer`

## PlainSFTTrainer

Undefended SFT baseline — no safety defense applied. Use this as the attack
target or comparison baseline when benchmarking other harden methods.

### Signature

```python
from safetune.runner import harden

trainer = harden.PlainSFTTrainer(model, tokenizer)
trainer.train(train_dataset, out_dir="./plain_sft_ckpt")
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Base model to fine-tune |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |

### Full example

```python
from safetune.runner import harden

trainer = harden.PlainSFTTrainer(model, tokenizer)
out_path = trainer.train(train_dataset, out_dir="./plain_sft_ckpt")
metrics  = trainer.eval("plain_sft", out_path)
trainer.save_results(metrics, variant="baseline")
```

### When to use

- **Best for:** establishing the undefended attack baseline for comparisons against harden methods.
- **Trade-offs:** no safety preservation — safety degrades when fine-tuned on contaminated data.

No citation: this is an undefended plain-SFT baseline, not a published method.

---

## SafeGradTrainer

### Signature

```python
# Runner API (recommended)
from safetune.runner import harden

trainer = harden.SafeGradTrainer(model, tokenizer, rho=1.0, kl_temperature=1.0)
trainer.train(train_dataset, safety_dataset=safety_dataset)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | The base model to fine-tune |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer for the model |
| `safety_dataset` | `Dataset` | `None` | Clean refusal / safety data (e.g. BeaverTails); passed to `.train(...)`. Built automatically if not supplied |

### SafeGradTrainer-specific parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `rho` | `float` | `1.0` | Gradient-surgery alignment weight: `g_final = g'_user + rho * g_align` |
| `kl_temperature` | `float` | `1.0` | KL alignment temperature |
| `reference_model_path` | `str` | `None` | HF path/ID of the frozen aligned reference for the KL signal; falls back to the tokenizer's model id |

### Full example

```python
from safetune.runner import harden

trainer = harden.SafeGradTrainer(model, tokenizer, rho=1.0, kl_temperature=1.0)
trainer.train(train_dataset, safety_dataset=safety_dataset)
```

### When to use

- **Best for:** keeping safety intact during a capability fine-tune (math, code, instruction-following) on an aligned model.
- **Trade-offs:** requires a second forward pass (safety batch) + a frozen reference model — roughly 2× memory vs plain SFT.

### Tips

- Start with `rho=0.1`; increase to `0.5` for stronger safety preservation at some capability cost.
- `reference_model_path` should point to the same aligned base you started from; it is loaded frozen. If omitted, the tokenizer's model id is used.

### Citation

```bibtex
@article{safegrad2025,
  title  = {Gradient Surgery for Safe LLM Fine-Tuning},
  year   = {2025},
  note   = {arXiv:2508.07172},
}
```
