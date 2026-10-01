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
(candidate layers, pooling, layer-selection sweep, KL threshold).

### RefusalDirectionConfig fields

| Field | Type | Default | Description |
|---|---|---|---|
| `target_layers` | `list[int] \| None` | `None` | Layers to extract candidates from; `None` is all decoder layers |
| `pick_layer` | `int \| None` | `None` | Force this layer and skip the sweep |
| `pool_method` | `str` | `"last_token"` | `"last_token"` (paper) or `"mean"` |
| `strength` | `float` | `1.0` | Steering / ablation multiplier |
| `normalize` | `bool` | `True` | L2-normalise the direction |
| `select_directions` | `bool` | `True` | Run the Arditi et al. validation sweep; `False` uses the middle layer |
| `kl_threshold` | `float` | `0.1` | Max KL (clean vs ablated, harmless prompts) for a candidate to survive |
| `induce_refusal_threshold` | `float` | `0.0` | Min induce score, when `require_induce=True` |
| `prune_layer_fraction` | `float` | `0.2` | Drop candidates in the last fraction of layers |
| `min_layer_fraction` | `float` | `0.2` | If the sweep's winner is in the first fraction of layers (`layer < int(n_layers * min_layer_fraction)`), or no candidate lowers the clean refusal rate, warn and fall back to the middle layer. `0` disables the floor |
| `n_val` | `int` | `16` | Harmful and harmless prompts held out for scoring |
| `max_new_tokens` | `int` | `24` | Tokens generated when scoring bypass |
| `require_induce` | `bool` | `False` | Also compute and enforce the induce score |

The sweep breaks ties toward the middle layer and logs the chosen layer at INFO
with its bypass score, KL and the clean refusal rate. Early layers carry
token-level features: on Tiny Aya the sweep without the `min_layer_fraction`
floor picked layer 1 of 36. If the sweep falls back or picks a poor layer, set
`pick_layer` by hand (e.g. 24 of 36 on Tiny Aya).

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
