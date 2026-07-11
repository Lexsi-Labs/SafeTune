# QReSafe — Quantization-aware safety patching

Safety patching designed for quantized models. Standard recovery methods apply
edits to float32 weights, which may be washed out after re-quantization.
QReSafe scores safety-critical weights with a SNIP saliency score and patches
only those. Two modes: `"selective"` keeps the safety-critical weights at full
precision while the rest are quantized, and `"lora_dpo"` trains a masked LoRA
adapter on safety-preference data post-quantization.

Based on "Q-resafe: Assessing Safety Risks and Quantization-aware Safety
Patching for Quantized Large Language Models" (Chen et al., ICML 2025,
arXiv:2506.20251).

## Signature

```python
QReSafeTrainer(
    model: nn.Module,
    *,
    mode: str = "selective",
    quant_bits: int = 4,
    calib_inputs: list | None = None,
    tau: float = 0.6,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Quantized model to patch |
| `mode` | `str` | `"selective"` | `"selective"` (keep safety-critical weights at full precision) or `"lora_dpo"` (masked LoRA adapter on safety-preference data) |
| `quant_bits` | `int` | `4` | Quantization bit-width |
| `calib_inputs` | `list \| None` | `None` | Tokenized calibration inputs used to compute the SNIP saliency score |
| `tau` | `float` | `0.6` | Saliency percentile threshold — weights above it are treated as safety-critical |

## Full example

```python
from safetune.runner import recover

trainer = recover.QReSafeTrainer(
    model,
    mode="selective",
    quant_bits=4,
    tau=0.6,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "qresafe_ckpt")
metrics = trainer.eval("qresafe_run", ckpt_path)
trainer.save_results(metrics, variant="selective_4bit")
```

## When to use

- **Best for:** 4-bit or 8-bit quantized models where standard float-weight edits are invalidated by re-quantization.
- **`"selective"` mode:** training-free; keeps the highest-saliency safety weights at full precision while the rest are quantized.
- **`"lora_dpo"` mode:** trains a masked LoRA adapter on safety-preference data after quantization — more thorough but requires the pre-quantization reference model and calibration prompts.
- **Compare to AAQ:** AAQ preserves alignment during the calibration step (before quantization); QReSafe restores it after.
