# SafeSteerTrainer — category-routed steering

Category-routed, training-free activation steering. The `SafeSteerModel` wrapper
maintains one steering vector per harm category and routes each prompt to the vector
for its detected category; each vector is a median-norm-pruned diff of activation
differences rather than a plain diff-of-means.

!!! note "The trainer uses a single default vector"
    `SafeSteerTrainer.calibrate` extracts one refusal direction and builds the wrapper
    with a single `{"default": vector}` and no classifier, so every prompt routes to
    that one vector. To use per-category routing, construct `SafeSteerModel` directly
    with a `category_vectors` dict and a `classifier` callable.

Ref: Ghosh et al., "SafeSteer: Interpretable Safety Steering with Refusal-Evasion in
LLMs," arXiv:2506.04250.

## Signature

```python
SafeSteerTrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    alpha: float = 15.0,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to steer |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `alpha` | `float` | `15.0` | Steering vector scaling coefficient |

For per-category routing, `SafeSteerModel` accepts `category_vectors`, a `classifier`
callable (`prompt_text -> category`), `layer_id`, `alpha`, and median-norm pruning
options (`prune`, `prune_quantile`).

## Full example

```python
from safetune.runner import steer

trainer = steer.SafeSteerTrainer(
    model, tokenizer,
    alpha=15.0,
)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

output = wrapped.generate(**tokenizer("How do I make a weapon?", return_tensors="pt"))
```

## When to use

- **Best for:** datasets that span multiple harm categories with different activation signatures, where a single global direction may be too coarse. Note the trainer builds a single default vector; pass a `category_vectors` dict and `classifier` to `SafeSteerModel` for true per-category routing.
- **Compare to CAA:** CAA uses one direction for all harm; SafeSteer's model supports one direction per category.

## Citation

```bibtex
@article{safesteer2025,
  title  = {SafeSteer: Interpretable Safety Steering with Refusal-Evasion in LLMs},
  author = {Ghosh and others},
  year   = {2025},
  note   = {arXiv:2506.04250},
}
```
