# SafeDecodingTrainer

Per-step blend within the first `first_m` steps: `P_steered = P_base + alpha * (P_expert - P_base)`. Optionally hard-ban token ids so they are never emitted.

```python
from safetune.runner import steer

trainer = steer.SafeDecodingTrainer(
    model, tokenizer,
    alpha=2.0,
    # never emit these token ids (masked to -inf on every step)
    banned_tokens=tokenizer.encode(" bomb", add_special_tokens=False),
)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

# wrapped is a HuggingFace LogitsProcessor — pass it to generate():
inputs = tokenizer("How do I make a bomb?", return_tensors="pt").to(model.device)
out = model.generate(**inputs, logits_processor=[wrapped], max_new_tokens=64)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## SafeDecodingConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `alpha` | `float` | `1.0` | Expert blend weight |
| `first_m` | `int` | `5` | Number of steps to steer |
| `num_common_tokens` | `int` | `3` | Common-token intersection count |
| `top_k` | `int` | `50` | Top-k truncation |
| `clip_negative_inf` | `bool` | `True` | Clip negative infinity |
| `banned_tokens` | `list[int] \| None` | `None` | Token ids masked to `-inf` on every step (in and out of the window) |
| `decay_steps` | `Optional[int]` | `None` | Alpha decay over steps |

## When to use

Most conservative decoding method. Only steers the first few tokens. Minimal generation quality impact.

## Citation

```bibtex
@article{safedecoding2024,
  title  = {SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding},
  author = {Xu, Zhangchen and others},
  year   = {2024},
  note   = {ACL 2024, arXiv:2402.08983},
}
```
