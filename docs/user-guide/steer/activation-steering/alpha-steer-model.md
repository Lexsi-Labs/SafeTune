# AlphaSteerTrainer — null-space-constrained ridge steering

Computes a per-layer steering transform by ridge-regularized regression of malicious
activations onto a refusal direction, constrained to the null space of benign
activations so harmless-prompt activations are left nearly unchanged. One matrix solve
per layer, no iterative optimisation.

Ref: "AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint," arXiv:2506.07022.

## Signature

```python
AlphaSteerTrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    alpha: float = 20.0,
    layers: list[int] | None = None,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to steer |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `alpha` | `float` | `20.0` | Global scaling (`strength`) on the computed steering matrix |
| `layers` | `list[int] \| None` | `None` | Target layers to steer; defaults to layers 10–19 if `None` |

## Full example

```python
from safetune.runner import steer

trainer = steer.AlphaSteerTrainer(
    model, tokenizer,
    alpha=20.0,
)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

output = wrapped.generate(**tokenizer("How do I make a bomb?", return_tensors="pt"))
```

## When to use

- **Best for:** steering without iterative gradient-based tuning — calibration is a per-layer matrix solve.
- **Null-space constraint** keeps benign activation patterns nearly unchanged by the steering matrix, which reduces over-refusal compared to plain CAA.
- **Compare to CAA:** AlphaSteer learns a per-layer matrix (not a single direction vector) and constrains it to the null space of benign activations. It requires solving a linear system per layer.

## Citation

```bibtex
@article{alphasteer2025,
  title  = {AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint},
  year   = {2025},
  note   = {arXiv:2506.07022},
}
```
