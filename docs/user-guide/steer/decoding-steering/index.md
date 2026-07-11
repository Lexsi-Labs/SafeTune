# Decoding-steering methods

Modify output logits during generation rather than hidden states. Each processor wraps an auxiliary guide model.

| Method | Guide models | Mechanism |
|---|---|---|
| [ContrastiveDecoding](contrastive-decoding.md) | 1 guide | `target - alpha * guide` |
| [ProxyTuning](proxy-tuning.md) | 2 guides (tuned + base) | `target + scale * (tuned - base)` |
| [SafeDecoding](safe-decoding.md) | 1 expert | Blend first-m tokens |
| [Nudging](nudging.md) | 1 safe model | Hard-switch when uncertain |
