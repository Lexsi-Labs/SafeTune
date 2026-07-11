# CAATrainer — Contrastive Activation Addition

Computes the difference of mean hidden states between paired refusal (positive)
and compliance (negative) prompts, then adds the resulting direction vector to the
residual stream at inference time.

Ref: Panickssery et al., "Steering Llama 2 via Contrastive Activation Addition,"
arXiv:2312.06681.

## Signature

```python
CAATrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    target_layers: list[int] | None = None,
    pool_method: str = "mean",
    normalize: bool = True,
    multiplier: float = 20.0,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to steer |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `target_layers` | `list[int] \| None` | `None` | Layers to steer; defaults to layers 14–18 if `None` |
| `pool_method` | `str` | `"mean"` | Hidden-state pooling: `"last_token"` or `"mean"` |
| `normalize` | `bool` | `True` | L2-normalise the CAA vector before applying |
| `multiplier` | `float` | `20.0` | Steering vector scaling coefficient |

## Full example

```python
from safetune.runner import steer

trainer = steer.CAATrainer(
    model, tokenizer,
    multiplier=20.0,
    normalize=True,
)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

output = wrapped.generate(**tokenizer("How do I make a weapon?", return_tensors="pt"))
wrapped.remove()  # remove hooks when done
```

## When to use

- **Best for:** contrastive steering when you have paired harmful / harmless prompts. The simplest direction-based baseline.
- **Tune `multiplier`:** higher values steer more aggressively but increase over-refusal.
- **Compare to RefusalDirectionTrainer:** RefusalDirection uses a single direction extracted from the model's own hidden states (no external pairs); CAA requires explicit paired examples but can encode more nuanced contrast.

## Citation

```bibtex
@article{caa2023,
  title  = {Steering Llama 2 via Contrastive Activation Addition},
  author = {Panickssery, Nina and Gabrieli, Nick and Schulz, Julian and Tong, Meg and Hubinger, Evan and Turner, Alexander Matt},
  year   = {2023},
  note   = {arXiv:2312.06681},
}
```
