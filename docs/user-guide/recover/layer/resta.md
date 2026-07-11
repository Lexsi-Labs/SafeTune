# RESTA: REstoring Safety through Task Arithmetic

Adds the alignment safety delta to the drifted model's weights:
$\theta_{\text{safe}} = \theta_{\text{finetuned}} + \text{alpha} \cdot (\theta_{\text{aligned}} - \theta_{\text{base}})$, with optional DARE sparsification
that drops and rescales elements of the delta to reduce interference with task capabilities.

Ref: Bhardwaj et al., "Language Models are Homer Simpson! Safety Re-Alignment of
Fine-tuned Language Models through Task Arithmetic," ACL 2024, arXiv:2402.11746.

## Signature

```python
ReStaTrainer(
    model: nn.Module,
    *,
    base_model: nn.Module,
    aligned_model: nn.Module,
    alpha: float = 1.0,
    dare: bool = True,
    dare_seed: int = 0,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `base_model` | `nn.Module` | required | Pre-alignment base model |
| `aligned_model` | `nn.Module` | required | Safety-aligned reference model |
| `alpha` | `float` | `1.0` | Safety delta multiplier |
| `dare` | `bool` | `True` | Apply DARE (drop-and-rescale) sparsification of the delta before adding (drop rate fixed at 0.9) |
| `dare_seed` | `int` | `0` | Seed for the DARE drop mask |

## Full example

```python
from safetune.runner import recover

trainer = recover.ReStaTrainer(
    model,
    base_model=base_model,
    aligned_model=aligned_model,
    alpha=1.0,
    dare=False,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "resta_ckpt")
metrics = trainer.eval("resta_run", ckpt_path)
trainer.save_results(metrics, variant="alpha=1.0")
```

## When to use

- **A layer-level recovery baseline.** It applies the full alignment delta; unlike [WiSE-FT](../whole-model/wise-ft.md), which interpolates, RESTA adds on top.
- **`dare=True` (default):** drop-and-rescale sparsification reduces task-capability interference when `alpha` is large.
- **Tune `alpha`:** values above `1.0` over-apply the safety delta (useful when drift is severe); values below `1.0` apply a partial patch.
- **Compare to [LoX](../low-rank/lox.md):** LoX keeps only the top-`rank` singular components of the delta; RESTA uses the full dense delta.
