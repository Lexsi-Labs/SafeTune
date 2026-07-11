# Pre-post merge

Linearly interpolates the drifted model toward its pre-fine-tuning checkpoint: the
snapshot saved before the fine-tuning run that caused the drift.

$W_{\text{merged}} = (1 - \alpha) \cdot W_{\text{drifted}} + \alpha \cdot W_{\text{pre}}$

This is the simplest possible weight-space recovery: no optimization, no reference
models, just a convex combination with the pre-FT weights.

> **Where does `pre_model` come from?** `pre_model` is the checkpoint you saved
> *before* running your fine-tune — typically via `model.save_pretrained("pre_ft_checkpoint/")`.
> If you did not save a pre-FT snapshot, use `task_arithmetic` or `apply_resta` instead
> (they reconstruct the safety task vector from a separate aligned reference model).

## Signature

```python
PrePostMergeTrainer(
    model: PreTrainedModel,
    *,
    pre_model: PreTrainedModel,
    alpha: float = 0.5,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | The drifted (post-fine-tuning) model to recover |
| `pre_model` | `PreTrainedModel` | required | The pre-fine-tuning checkpoint saved before the drift-causing run |
| `alpha` | `float` | `0.5` | Interpolation weight; `0.0` = drifted unchanged, `1.0` = fully reverts to pre-FT |

## Full example

```python
from transformers import AutoModelForCausalLM
from safetune.runner import recover

# Load the pre-fine-tuning checkpoint you saved before training
pre_ft_model = AutoModelForCausalLM.from_pretrained("./pre_ft_checkpoint")

trainer = recover.PrePostMergeTrainer(model, pre_model=pre_ft_model, alpha=0.5)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "prepost_merge_ckpt")
metrics = trainer.eval("prepost_merge_run", ckpt_path)
trainer.save_results(metrics, variant="alpha=0.5")
```

**Sweep `alpha` to tune the safety/capability trade-off:**

```python
for alpha in [0.2, 0.4, 0.6, 0.8]:
    patched = recover.PrePostMergeTrainer(model, pre_model=pre_ft_model, alpha=alpha).apply()
    result = evaluate(patched, benchmarks=["harmbench"])
    print(f"alpha={alpha}  safety={result['harmbench']['refusal_rate']:.2%}")
```

## When to use

- **Best for:** situations where you saved the pre-FT checkpoint and want the simplest possible recovery with zero compute.
- **Not for:** cases where no pre-FT checkpoint exists — use `apply_resta` or `task_arithmetic` instead.
- **Trade-offs:** higher `alpha` restores more safety but also undoes more fine-tuning (capability regression). Sweep `alpha` on a small eval set.

## Citation

```bibtex
@article{farn2024prepost,
  title  = {Safeguard Fine-Tuned LLMs Through Pre- and Post-Tuning Model Merging},
  author = {Farn, Hua and Su, Hsuan and Kumar, Shachi H and Sahay, Saurav and Chen, Shang-Tse and Lee, Hung-yi},
  year   = {2024},
  note   = {arXiv:2412.19512},
}
```
