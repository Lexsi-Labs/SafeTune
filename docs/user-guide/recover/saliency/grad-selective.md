# Gradient selective recovery

Computes per-weight gradient saliency by running a forward+backward pass on
harmful calibration data. The `top_fraction` of weights with the highest
saliency (most responsible for unsafe outputs) are then restored from the
aligned reference model, while the remaining weights are left unchanged.

## Signature

```python
GradSelectiveRecoverTrainer(
    model: nn.Module,
    *,
    aligned_model: nn.Module,
    harmful_inputs: Sequence[torch.Tensor],
    top_fraction: float = 0.1,
    max_samples: int = 32,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `aligned_model` | `nn.Module` | required | Safety-aligned reference model to restore values from |
| `harmful_inputs` | `Sequence[Tensor]` | required | Sequence of tokenized `input_ids` tensors (shape `(1, T)` each) from harmful prompts |
| `top_fraction` | `float` | `0.1` | Fraction of highest-saliency weights per parameter to restore |
| `max_samples` | `int` | `32` | Maximum number of harmful calibration samples used |

## Full example

```python
from safetune.runner import recover

harmful_inputs = [
    tokenizer(p, return_tensors="pt").input_ids
    for p in harmful_prompts
]

trainer = recover.GradSelectiveRecoverTrainer(
    model,
    aligned_model=aligned_model,
    harmful_inputs=harmful_inputs,
    top_fraction=0.1,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "grad_selective_ckpt")
metrics = trainer.eval("grad_selective_run", ckpt_path)
trainer.save_results(metrics, variant="top_frac=0.1")
```

## When to use

- **Best for:** targeted recovery when you want to use the model's own gradients (not just weight magnitudes) to identify what changed during drift.
- **Compare to Antidote v1:** Antidote uses WANDA-style `|W| × ||X||_2` activation importance; this method uses actual loss gradients on the harmful `input_ids` — more accurate but costs a backward pass per sample.
- **`top_fraction`:** start at `0.05`; increase if safety is not restored.
