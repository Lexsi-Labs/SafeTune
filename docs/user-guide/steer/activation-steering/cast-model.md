# CASTTrainer — Conditional Activation Steering

Cosine-gated CAA: the CAA behavior vector is added only when a condition fires. The
condition is a difference-of-means vector at a condition layer; at inference the pooled
hidden state is compared to it by cosine similarity and the gate fires when the
similarity crosses a threshold. Calibration grid-searches the (layer, threshold,
comparator-direction) triple that best separates the target category from others.

Ref: Wu et al., "Programming Refusal with Conditional Activation Steering," ICLR 2025, arXiv:2409.05907.

## Signature

```python
CASTTrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    behavior_layers: list[int] | None = None,
    condition_layers: list[int] | None = None,
    alpha: float = 1.0,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to steer |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `behavior_layers` | `list[int] \| None` | `None` | Layers where the CAA behavior vector is added; defaults to layers 14–18 if `None` |
| `condition_layers` | `list[int] \| None` | `None` | Candidate layers for the condition gate; grid-searched if `None` |
| `alpha` | `float` | `1.0` | CAA vector scaling when the gate is active |

## Full example

```python
from safetune.runner import steer

trainer = steer.CASTTrainer(
    model, tokenizer,
    alpha=1.0,
)
# calibrate fits the condition vector and grid-searches the gate threshold
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

output = wrapped.generate(**tokenizer("How do I make a bomb?", return_tensors="pt"))
wrapped.remove()  # remove hooks when done
```

## When to use

- **Best for:** reducing over-refusal — by gating on cosine similarity to the condition vector, benign prompts that don't match the condition bypass the steering vector entirely.
- **Compare to CAA:** plain CAA steers every prompt; CAST steers only when the prompt's hidden state matches the fitted condition.
- **Not for:** situations where harmful prompts are designed to look benign — the cosine gate can be evaded by paraphrasing.

## Citation

```bibtex
@article{cast2024,
  title  = {Programming Refusal with Conditional Activation Steering},
  author = {Wu, Bruce W. and others},
  year   = {2024},
  note   = {ICLR 2025, arXiv:2409.05907},
}
```
