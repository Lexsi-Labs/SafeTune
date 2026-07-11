# WiSE-FT — Weight-space interpolation

Linear interpolation between the drifted model and the aligned model:
$\theta_{\text{recovered}} = (1 - \text{alpha}) \cdot \theta_{\text{drifted}} + \text{alpha} \cdot \theta_{\text{aligned}}$.
The standard robustness baseline from the WiSE-FT paper.

Ref: Wortsman et al., "Robust Fine-Tuning of Zero-Shot Models," CVPR 2022.

## Signature

```python
WiseFTTrainer(
    model: nn.Module,
    *,
    aligned_model: nn.Module,
    alpha: float = 0.5,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `aligned_model` | `nn.Module` | required | Safety-aligned reference model |
| `alpha` | `float` | `0.5` | Interpolation coefficient toward aligned; `0.0` = no recovery, `1.0` = fully replaces with aligned |

## Full example

```python
from safetune.runner import recover

trainer = recover.WiseFTTrainer(model, aligned_model=aligned_model, alpha=0.5)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "wiseft_ckpt")
metrics = trainer.eval("wiseft_run", ckpt_path)
trainer.save_results(metrics, variant="alpha=0.5")
```

## When to use

- **Best for:** safety-utility trade-off; `alpha=0.5` gives an equal blend of the drifted and aligned weights.
- **Sweep `alpha`:** try `[0.1, 0.3, 0.5, 0.7, 0.9]` and pick the knee of the safety/capability trade-off curve.
- **Compare to task arithmetic:** WiSE-FT interpolates $\theta_{\text{drifted}}$ toward $\theta_{\text{aligned}}$; task arithmetic adds the delta $\theta_{\text{aligned}} - \theta_{\text{base}}$. The two only coincide under specific weight relationships (e.g. when $\theta_{\text{base}}$ equals $\theta_{\text{drifted}}$), not in general.

## Citation

```bibtex
@article{wiseft2022,
  title  = {Robust Fine-Tuning of Zero-Shot Models},
  author = {Wortsman, Mitchell and others},
  year   = {2022},
  note   = {CVPR 2022, arXiv:2109.01903},
}
```
