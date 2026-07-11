# Layer-level recovery

Apply safety corrections at per-layer granularity.

| Method | Mechanism |
|---|---|
| [SafeMerge](safe-merge.md) | Per-layer cosine gating |
| [Safe LoRA](safe-lora.md) | Project unsafe LoRA onto safe subspace |
| [RESTA](resta.md) | Add safety vector with optional DARE sparsification |
| [Safe Delta](safe-delta.md) | OBS-style entry-wise weight edit |
| [QReSafe](qresafe.md) | Quantization-aware safety patching |
| [AAQ](aaq.md) | Alignment-aware quantization |
| [RepNoise recover](repnoise-recover.md) | Post-hoc RepNoise injection |
