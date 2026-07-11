# SafeMERGE: per-layer subspace gating

Per weight matrix, builds a Safe-LoRA-style safety subspace projection matrix from
the aligned-minus-base delta, then measures the cosine similarity between the
fine-tuning task delta (`W_ft − W_base`) and its projection onto that subspace.
Layers whose cosine falls below `threshold` are considered "unsafe" and are merged
toward the aligned model by `alpha`; layers above the threshold keep the fine-tuned
weights untouched.

Ref: Djuhera et al., "SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large
Language Models via Selective Layer-Wise Model Merging," ICLR 2025 Workshop,
arXiv:2503.17239.

## Signature

```python
SafeMergeTrainer(
    model: nn.Module,
    *,
    base_model: nn.Module,
    aligned_model: nn.Module,
    threshold: float = 0.35,
    alpha: float = 0.5,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `base_model` | `nn.Module` | required | Pre-alignment base model (used to compute alignment delta per layer) |
| `aligned_model` | `nn.Module` | required | Safety-aligned reference model |
| `threshold` | `float` | `0.35` | Cutoff on the cosine between the task delta and its safety-subspace projection; layers below this are merged |
| `alpha` | `float` | `0.5` | Merge strength for drifted layers: `(1-alpha) * drifted + alpha * aligned` |

## Full example

```python
from safetune.runner import recover

trainer = recover.SafeMergeTrainer(
    model,
    base_model=base_model,
    aligned_model=aligned_model,
    threshold=0.35,
    alpha=0.5,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "safemerge_ckpt")
metrics = trainer.eval("safemerge_run", ckpt_path)
trainer.save_results(metrics, variant="threshold=0.35")
```

## When to use

- **Best for:** selective recovery — fine-tuning typically drifts only a subset of layers; merging all layers wastes capability.
- **Tune `threshold`:** lower = merge more layers; higher = only merge the most severely drifted.
- **Compare to WiSE-FT:** WiSE-FT merges all layers at the same `alpha`; SafeMerge is layer-selective.
