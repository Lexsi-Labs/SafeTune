# Task arithmetic

Add the safety task vector $v = \text{aligned} - \text{base}$ to the drifted model:
$\theta_{\text{recovered}} = \theta_{\text{drifted}} + \text{alpha} \cdot (\theta_{\text{aligned}} - \theta_{\text{base}})$.

Ref: Ilharco et al., "Editing Models with Task Arithmetic," ICLR 2023.

## Signature

```python
TaskArithmeticTrainer(
    model: nn.Module,
    *,
    base_model: nn.Module,
    aligned_model: nn.Module,
    alpha: float = 1.0,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | The post-fine-tune (drifted) model — modified in-place |
| `base_model` | `nn.Module` | required | Pre-alignment base model |
| `aligned_model` | `nn.Module` | required | Safety-aligned reference model |
| `alpha` | `float` | `1.0` | Task vector multiplier; `alpha > 1.0` over-applies the safety delta |

## Full example

```python
from safetune.runner import recover

trainer = recover.TaskArithmeticTrainer(
    model, base_model=base_model, aligned_model=aligned_model, alpha=1.0,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "task_arithmetic_ckpt")
metrics = trainer.eval("task_arithmetic_run", ckpt_path)
trainer.save_results(metrics, variant="alpha=1.0")
```

## When to use

- **Start here:** simplest whole-model recovery baseline; no hyperparameters to tune beyond `alpha`.
- **Tune `alpha`:** `1.0` adds the full safety delta; sweep `[0.5, 1.0, 1.5, 2.0]` on a small eval set to find the best value.
- **Compare to WiSE-FT:** WiSE-FT interpolates between two models and task arithmetic adds a delta vector. Adding the delta $\theta_{\text{aligned}} - \theta_{\text{base}}$ to $\theta_{\text{drifted}}$ is not in general the same as interpolating $\theta_{\text{drifted}}$ toward $\theta_{\text{aligned}}$, so the two only coincide under specific weight relationships.

## Citation

```bibtex
@article{taskarithmetic2023,
  title  = {Editing Models with Task Arithmetic},
  author = {Ilharco, Gabriel and others},
  year   = {2023},
  note   = {ICLR 2023, arXiv:2212.04089},
}
```
