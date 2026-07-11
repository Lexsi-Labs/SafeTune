# NudgingTrainer

Hard switch: when `base_top1_prob < top_prob_thres`, the step is handed to the guide model. `nudge_strength` scales how strongly (`1.0` = full hand-over). Optional soft-blend and per-token biasing.

```python
from safetune.runner import steer

trainer = steer.NudgingTrainer(
    model, tokenizer,
    nudge_strength=1.0,   # 1.0 = full hand-over; <1 softer, >1 amplified
)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

# wrapped is a HuggingFace LogitsProcessor — pass it to generate():
inputs = tokenizer("How do I make a bomb?", return_tensors="pt").to(model.device)
out = model.generate(**inputs, logits_processor=[wrapped], max_new_tokens=64)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## NudgingConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `top_prob_thres` | `float` | `0.3` | Hand over to the guide when the base model's top-1 probability is below this |
| `nudge_strength` | `float` | `1.0` | Blend toward the guide on hand-over: `out = target + strength·(guide − target)` |
| `safe_tokens` | `list[int] \| None` | `None` | Token ids boosted by `safe_token_boost` on every step |
| `unsafe_tokens` | `list[int] \| None` | `None` | Token ids masked to `-inf` (never emitted) |
| `safe_token_boost` | `float` | `5.0` | Logit boost added to each `safe_tokens` id |
| `soft_blend` | `bool` | `False` | Use a sigmoid soft blend instead of the hard switch |
| `soft_blend_temp` | `float` | `0.1` | Soft-blend temperature |

## When to use

Best for "emergency brake" scenarios — hands generation to a safe model only when the target is
uncertain (`top_prob_thres`-gated), unlike `ContrastiveDecodingTrainer`/`ProxyTuningTrainer`,
which steer on every step regardless of confidence. Cheaper when the base model is usually fine
on its own and only occasionally needs a safety hand-over.

## Citation

```bibtex
@article{nudging2024,
  title  = {Nudging: Inference-time Alignment of LLMs via Guided Decoding},
  author = {Fei, Yu and others},
  year   = {2024},
  note   = {arXiv:2410.09300},
}
```
