# RefusalDirectionTrainer

Extract a single refusal direction and steer (add) or ablate (project out) at inference time.

Ref: Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction,"
NeurIPS 2024, arXiv:2406.11717.

## Signature

```python
RefusalDirectionTrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    layers: list[int] | None = None,
    alpha: float = 20.0,
    orthogonalize: bool = False,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to steer |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `layers` | `list[int] \| None` | `None` | Layers the steering hook writes into; all decoder layers if `None` |
| `alpha` | `float` | `20.0` | Steering coefficient added to the residual stream |
| `orthogonalize` | `bool` | `False` | Reserved flag for weight-space orthogonalisation |

`calibrate` extracts the direction with `extract_refusal_direction` and wraps the
model in `RefusalDirectionModel` in `"steer"` mode. You can override the coefficient
per call with `calibrate(..., alpha=...)`.

Direction extraction is configured separately through `RefusalDirectionConfig`
(candidate layers, pooling, layer-selection sweep, KL threshold). See the
weight-space example below for its fields.

## Modes

The wrapper `RefusalDirectionModel` supports two hook modes:

| Mode | Effect |
|---|---|
| `"steer"` | `h ← h + strength · direction` |
| `"ablate"` | `h ← h − (h · direction) · direction` |

`RefusalDirectionTrainer.calibrate` always builds the wrapper in `"steer"` mode.
To ablate, construct `RefusalDirectionModel(..., mode="ablate")` directly (see below).

## Full example

```python
from safetune.runner import steer

# Steer mode — add refusal direction
trainer = steer.RefusalDirectionTrainer(
    model, tokenizer,
    alpha=20.0,
)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)
wrapped.install()
output = wrapped.model.generate(**tokenizer("How do I make a bomb?", return_tensors="pt"))
wrapped.remove()
```

For ablate mode (project out the refusal direction, e.g. for red-teaming), build the
wrapper directly — the trainer's `calibrate` only produces `"steer"` wrappers:

```python
from safetune.steer import extract_refusal_direction, RefusalDirectionModel

direction, _, _ = extract_refusal_direction(
    model, tokenizer, harmful_prompts, harmless_prompts,
)
ablated = RefusalDirectionModel(model, direction=direction, mode="ablate")
ablated.install()
```

## Weight-space materialisation

```python
import torch
from safetune.steer import (
    extract_refusal_direction,
    RefusalDirectionConfig,
    orthogonalize_weights,
    restore_weights,
)

harmful_prompts = ["How do I make a bomb?", "Explain how to hack a bank."]
harmless_prompts = ["How do I bake bread?", "Explain how photosynthesis works."]

# Derive the refusal direction: mean(harmful) − mean(harmless) last-token
# hidden states at a middle layer, L2-normalised to a (hidden_size,) unit vector.
# select_directions=False skips the slow validation sweep (middle-layer heuristic).
direction, layer_idx, _ = extract_refusal_direction(
    model, tokenizer, harmful_prompts, harmless_prompts,
    RefusalDirectionConfig(select_directions=False),
)
assert direction.shape == (model.config.hidden_size,)

# Permanently remove the refusal direction from weights (no runtime hooks).
# Returns a snapshot dict mapping each edited projection to its original tensor.
snapshots = orthogonalize_weights(model, direction)

# Undo — copies the saved originals back into every edited projection.
restore_weights(model, snapshots)
```

## When to use

- **Start here:** simplest and most well-understood steering method.
- **`"steer"` mode:** reinforces refusal at inference without modifying weights.
- **`"ablate"` mode:** use for red-teaming — removes refusal capability to measure robustness.
- **Weight-space materialisation:** use `orthogonalize_weights` for a permanent, zero-overhead version that requires no inference hooks.

## Citation

```bibtex
@article{refusaldirection2024,
  title  = {Refusal in Language Models Is Mediated by a Single Direction},
  author = {Arditi, Andy and others},
  year   = {2024},
  note   = {NeurIPS 2024, arXiv:2406.11717},
}
```
