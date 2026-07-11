# ContrastiveDecodingTrainer

Single-guide contrastive decoding: `out = target - alpha * guide` with adaptive plausibility masking.

```python
from safetune.runner import steer

trainer = steer.ContrastiveDecodingTrainer(model, tokenizer, alpha=1.0)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

# wrapped is a HuggingFace LogitsProcessor — pass it to generate():
inputs = tokenizer("How do I make a bomb?", return_tensors="pt").to(model.device)
out = model.generate(**inputs, logits_processor=[wrapped], max_new_tokens=64)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## ContrastiveDecodingConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `alpha` | `float` | `0.5` | Contrast weight |
| `adaptive_eps` | `float` | `0.1` | Adaptive plausibility epsilon |

## When to use

Fast, single-guide baseline: subtracts a weighted penalty for the guide's preferred tokens on
*every* decoding step, unlike `NudgingTrainer` (only intervenes when the base model is
uncertain) or `SafeDecodingTrainer` (only steers the first `first_m` tokens). Needs a single
guide model rather than `ProxyTuningTrainer`'s tuned+base pair — simplest setup among the
four decoding methods, at the cost of steering unconditionally throughout generation.

## Citation

```bibtex
@article{cd2022,
  title  = {Contrastive Decoding: Open-ended Text Generation as Optimization},
  author = {Li, Xiang Lisa and others},
  year   = {2022},
  note   = {ACL 2023, arXiv:2210.15097},
}
```
