# AAQ — Alignment-Aware Quantization

Alignment-aware quantization that preserves safety alignment through the
quantization process. Rather than fine-tuning the model, AAQ optimizes a small
set of pre-quantization transformation parameters with the Alignment-Preserving
Contrastive (APC) loss — a top-K KL-divergence objective on the output logit
distributions of the aligned and base reference models — and then applies the
quantizer. The APC loss pulls the quantized model toward the aligned model and
pushes it away from the base model on the tokens where the two disagree.

Based on "Alignment-Aware Quantization for LLM Safety" (Wee et al.,
arXiv:2511.07842).

## Signature

```python
AAQTrainer(
    model: nn.Module,
    *,
    aligned_model_path: str,
    base_model_path: str,
    calibration_steps: int = 10,
    lr: float = 5e-6,
    probe_texts: list[str] | None = None,
    simulate_quantization: bool = True,
    apc_weight: float = 0.1,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Float model to quantize with alignment-preserving calibration |
| `aligned_model_path` | `str` | required | HF path or local dir of the aligned model (positive reference) |
| `base_model_path` | `str` | required | HF path or local dir of the base model (negative reference) |
| `calibration_steps` | `int` | `10` | Number of APC calibration steps |
| `lr` | `float` | `5e-6` | APC calibration learning rate |
| `probe_texts` | `list[str] \| None` | `None` | Calibration probe texts (unlabelled) |
| `simulate_quantization` | `bool` | `True` | Apply simulated quantization noise during calibration |
| `apc_weight` | `float` | `0.1` | Weight of the contrastive term in the APC loss |

## Full example

```python
from safetune.runner import recover

trainer = recover.AAQTrainer(
    model,
    aligned_model_path="./aligned",
    base_model_path="./base",
    calibration_steps=10,
    apc_weight=0.1,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "aaq_ckpt")
metrics = trainer.eval("aaq_run", ckpt_path)
trainer.save_results(metrics, variant="4bit")
```

## When to use

- **Best for:** quantizing an aligned model while ensuring refusal is preserved — apply instead of standard PTQ.
- **Precedes QReSafe:** AAQ is used during quantization; QReSafe is used to patch an already-quantized model that lost safety.
- **Compare to QReSafe:** AAQ is proactive (alignment-aware calibration); QReSafe is reactive (post-quantization safety repair).
