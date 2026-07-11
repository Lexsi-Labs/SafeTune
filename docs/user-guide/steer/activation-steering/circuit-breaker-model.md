# CircuitBreakerTrainer — representation rerouting

Training-time method built around a retain + rerouting (RR) objective. The rerouting loss pushes the model's hidden representations on harmful inputs away from their original directions, while the retain loss keeps benign representations close to the frozen original model.

**Loss:** $L = c_{\text{retain}} \cdot \|h_{\text{lora, retain}} - h_{\text{orig, retain}}\|_2 + c_{\text{cb}} \cdot \mathrm{ReLU}(\cos(h_{\text{lora, cb}}, h_{\text{orig, cb}}))$, with a linear schedule $c_{\text{retain}} = \alpha \cdot (\text{step}/T)$ and $c_{\text{cb}} = \alpha \cdot (1 - \text{step}/T)$.

> **Note:** This is a training-time method — it modifies weights. It appears under activation-steering because it targets hidden representations. `calibrate()` returns a `CircuitBreakerModel` wrapper that exposes the RR loss (`compute_rr_loss`) and coefficient schedule (`rr_coefficients`) so you can run the fine-tuning loop yourself; it does not train inside `calibrate` and installs no inference-time hook by default.

Ref: Zou et al., "Improving Alignment and Robustness with Circuit Breakers," NeurIPS 2024, arXiv:2406.04313.

## Signature

```python
CircuitBreakerTrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None = None,
)
```

`CircuitBreakerTrainer` takes no method-specific keyword arguments; `calibrate()`
wraps the model in `CircuitBreakerModel`, whose RR hyperparameters
(`threshold`, `rr_alpha`, `rr_schedule_steps`, `target_layers`) are set on that
model, not on the trainer.

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to train |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |

## Full example

```python
from safetune.runner import steer

# harmful_prompts / harmless_prompts — contrast examples (list of str)
trainer = steer.CircuitBreakerTrainer(model, tokenizer)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

# `wrapped` is a CircuitBreakerModel exposing the RR objective. Run your own
# fine-tuning loop with wrapped.compute_rr_loss(...) and the coefficient
# schedule from wrapped.rr_coefficients(step) to update the model weights.
```

## When to use

- **Best for:** weight-level rerouting when you want circuit-breaker safety baked into the model itself rather than an inference-time hook.
- **Not for:** situations where you need a reversible inference-time intervention — use `RefusalDirectionTrainer` or `CAATrainer` instead.
- **Trade-offs:** requires a training run; you supply the optimizer loop around the RR loss.

## Citation

```bibtex
@article{circuitbreaker2024,
  title  = {Improving Alignment and Robustness with Circuit Breakers},
  author = {Zou, Andy and others},
  year   = {2024},
  note   = {NeurIPS 2024, arXiv:2406.04313},
}
```
