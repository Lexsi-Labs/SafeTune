# AdaSteerTrainer — adaptive two-direction steering

Learns a Rejection Direction (RD) and a Harmfulness Direction (HD), then fits a
per-input logistic regression on their projections to compute adaptive steering
coefficients. Unlike fixed-strength methods, the intervention scales with how
harmful a prompt appears to be.

Ref: Zhao et al., "AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender," EMNLP 2025, arXiv:2504.09466.

## Signature

```python
AdaSteerTrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    alpha: float = 15.0,
    layers: list[int] | None = None,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to steer |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `alpha` | `float` | `15.0` | Base steering multiplier passed to the AdaSteer model |
| `layers` | `list[int] \| None` | `None` | Target layers to steer; auto-selected if `None` |

## Full example

```python
from safetune.runner import steer

trainer = steer.AdaSteerTrainer(
    model, tokenizer,
    alpha=15.0,
)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

output = wrapped.generate(**tokenizer("How do I make a bomb?", return_tensors="pt"))
```

## When to use

- **Best for:** prompts where harmfulness varies widely in degree — the logistic regression adjusts the steering coefficient per-input rather than applying a fixed multiplier.
- **Compare to CAA:** CAA uses a fixed multiplier; AdaSteer adapts it per-prompt by projecting onto RD and HD and scoring with a learned regression.
- **Needs:** enough harmful and harmless calibration examples for the logistic regression to generalise.

## Citation

```bibtex
@article{adasteer2025,
  title  = {AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender},
  author = {Zhao, Weixiang and others},
  year   = {2025},
  note   = {EMNLP 2025, arXiv:2504.09466},
}
```
