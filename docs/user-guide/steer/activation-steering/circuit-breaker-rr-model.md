# CircuitBreakerRRTrainer — dual-purpose RR

!!! warning "Training-time method"
    `CircuitBreakerRRTrainer` requires a training loop (RR loss) and produces a
    trained checkpoint, not an inference-only wrapper. It is placed under Steer
    because its mechanism is representation rerouting.

Representation rerouting (RR) with a configurable threshold. `calibrate()` extracts
refusal directions from the contrast set, builds a `CircuitBreakerRRModel`, and calls
`.install()` so the RR behaviour is applied through runtime hooks. The model also exposes
the RR loss (`rr_loss`) and coefficient schedule (`rr_coeffs`) for training-time use.

Ref: Zou et al., "Improving Alignment and Robustness with Circuit Breakers," NeurIPS 2024, arXiv:2406.04313.

## Signature

```python
CircuitBreakerRRTrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    rr_layers: list[int] | None = None,
    target_layers: list[int] | None = None,
    threshold: float = 0.5,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to train |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `rr_layers` | `list[int] \| None` | `None` | Layers used for the RR loss during training |
| `target_layers` | `list[int] \| None` | `None` | Layers where the runtime reroute hooks are installed |
| `threshold` | `float` | `0.5` | Reroute threshold passed to `CircuitBreakerRRConfig` |

The remaining RR knobs (`strength`, `reroute_to`, `alpha`, `schedule_steps`) live on
`CircuitBreakerRRConfig` / `CircuitBreakerRRModel`, not on the trainer, and keep their
config defaults (`strength=1.0`, `reroute_to="zero"`, `alpha=5.0`, `schedule_steps=300`).

## Full example

```python
from safetune.runner import steer

trainer = steer.CircuitBreakerRRTrainer(
    model, tokenizer,
    threshold=0.5,
)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)
# calibrate() calls wrapped.install(); call wrapped.remove() to detach the hooks
```

## When to use

- **Best for:** representation rerouting with runtime hooks and access to the RR loss in one class.
- **Compare to CircuitBreakerTrainer:** `CircuitBreakerTrainer` exposes the two-term retain + rerouting loss from the original paper; this variant adds a threshold gate and, through `CircuitBreakerRRConfig`, a `reroute_to` destination.
- **`reroute_to="zero"`** — collapses harmful activations to zero (hard suppression).
- **`reroute_to="noise"`** — redirects to random noise (softer).

## Citation

```bibtex
@article{circuitbreaker2024,
  title  = {Improving Alignment and Robustness with Circuit Breakers},
  author = {Zou, Andy and others},
  year   = {2024},
  note   = {NeurIPS 2024, arXiv:2406.04313},
}
```
