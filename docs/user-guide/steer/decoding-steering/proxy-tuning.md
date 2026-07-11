# ProxyTuningTrainer

Two-model processor: `delta = scale * (tuned_logits - base_logits)`, `out = target_logits + delta`.

```python
from safetune.runner import steer

trainer = steer.ProxyTuningTrainer(model, tokenizer, proxy_model=aligned_model)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

# wrapped is a HuggingFace LogitsProcessor — pass it to generate():
inputs = tokenizer("How do I make a bomb?", return_tensors="pt").to(model.device)
out = model.generate(**inputs, logits_processor=[wrapped], max_new_tokens=64)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## ProxyTuningConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `scale` | `float` | `1.0` | Delta scaling factor |
| `clamp_delta` | `Optional[float]` | `None` | Max absolute delta |

## When to use

Requires two guide models (tuned + base). Can add safety-specific logit patterns. Closer to weight-space editing.

## Citation

```bibtex
@article{proxytuning2024,
  title  = {Tuning Language Models by Proxy},
  author = {Liu, Alisa and others},
  year   = {2024},
  note   = {arXiv:2401.08565},
}
```
