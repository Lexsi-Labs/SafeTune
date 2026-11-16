# CircuitBreakerRRTrainer — dual-purpose RR

!!! warning "The trainer installs runtime hooks; it does not train"
    `CircuitBreakerRRTrainer.calibrate()` extracts refusal directions, wraps the model in
    `CircuitBreakerRRModel`, installs forward hooks and returns the wrapper. It runs no
    training loop, changes no weights and writes no checkpoint. The hooks are a
    training-free approximation of circuit breakers, not the paper's method. The paper's
    method is the RR / LoRRA loss, exposed as `CircuitBreakerRRModel.rr_loss` (with the
    `rr_coeffs` schedule) for you to optimise inside your own LoRA fine-tuning loop.

!!! warning "Without `baselines`, results depend on call history"
    The runtime hook z-scores each position's projection against running statistics that accumulate across every
    forward pass (and so across prompts, batch order, padding side and cache use) until `remove()`. Pass
    `baselines={layer: (mean, std)}` for a history-free hook; the trainer path has none and warns once.

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
| `model` | `PreTrainedModel` | required | Model to hook; its weights are not changed |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `rr_layers` | `list[int] \| None` | `None` | Accepted and ignored (warns): the trainer trains nothing |
| `target_layers` | `list[int] \| None` | `None` | Accepted and ignored (warns): the hooks go on every layer that has a refusal direction |
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

- **Best for:** a training-free, threshold-gated runtime approximation of representation rerouting, with the RR training loss available on the same class for your own fine-tuning loop.
- **Compare to CircuitBreakerTrainer:** `CircuitBreakerTrainer` exposes the two-term retain + rerouting loss from the original paper; this variant adds a threshold gate and, through `CircuitBreakerRRConfig`, a `reroute_to` destination.
- **`reroute_to="zero"`** (default): on flagged tokens, subtracts `strength` times the component along the refusal direction (`strength=1.0` removes it).
- **`reroute_to="orthogonal"`**: replaces flagged tokens with their projection onto the orthogonal complement of the refusal direction, ignoring `strength`.
- Any other `reroute_to` value raises `ValueError` on the first forward pass.

## Citation

```bibtex
@article{circuitbreaker2024,
  title  = {Improving Alignment and Robustness with Circuit Breakers},
  author = {Zou, Andy and others},
  year   = {2024},
  note   = {NeurIPS 2024, arXiv:2406.04313},
}
```
