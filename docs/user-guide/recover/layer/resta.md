# RESTA: REstoring Safety through Task Arithmetic

Adds the alignment safety delta to the drifted model's weights:
$\theta_{\text{safe}} = \theta_{\text{finetuned}} + \text{alpha} \cdot (\theta_{\text{aligned}} - \theta_{\text{base}})$, with optional DARE sparsification
that drops and rescales elements of the delta to reduce interference with task capabilities.

Ref: Bhardwaj et al., "Language Models are Homer Simpson! Safety Re-Alignment of
Fine-tuned Language Models through Task Arithmetic," ACL 2024, arXiv:2402.11746.

## Signature

```python
ReStaTrainer(
    model: nn.Module,
    *,
    base_model: nn.Module,
    aligned_model: nn.Module,
    alpha: float = 1.0,
    dare: bool = True,
    dare_drop_rate: float | None = None,
    dare_seed: int = 0,
    device: str | torch.device | None = None,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `base_model` | `nn.Module` | required | Pre-alignment base model |
| `aligned_model` | `nn.Module` | required | Safety-aligned reference model |
| `alpha` | `float` | `1.0` | Safety delta multiplier |
| `dare` | `bool` | `True` | Apply DARE (drop-and-rescale) sparsification of the delta before adding |
| `dare_drop_rate` | `float \| None` | `None` | DARE drop probability p; `None` is 0.3, the RESTA paper's value (was 0.9). `dare_drop_rate=0.9` or `safetune.configure(legacy_resta_drop_rate=True)` restores the old rate. `apply_resta(dare_drop_rate=None)` uses the same default; `apply_resta` keeps `dare=False` by default |
| `dare_seed` | `int` | `0` | Seed for the DARE drop mask |
| `device` | `str \| torch.device \| None` | `None` | Where each per-tensor delta is computed. `None`: the drifted model's weight device. `"cpu"` keeps the extra memory off the GPU (use it with `base` / `aligned` loaded on CPU) |

## Full example

```python
from safetune.runner import recover

trainer = recover.ReStaTrainer(
    model,
    base_model=base_model,
    aligned_model=aligned_model,
    alpha=1.0,
    dare=False,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "resta_ckpt")
metrics = trainer.eval("resta_run", ckpt_path)
trainer.save_results(metrics, variant="alpha=1.0")
```

## When to use

- **A layer-level recovery baseline.** It applies the full alignment delta; unlike [WiSE-FT](../whole-model/wise-ft.md), which interpolates, RESTA adds on top.
- **`dare=True` (default):** drop-and-rescale sparsification reduces task-capability interference when `alpha` is large.
- **Small models:** DARE drops a fraction p of the delta's entries and scales the rest by 1/(1-p). On Qwen2.5-0.5B, with the full base-to-instruct delta as the safety vector, p=0.9 broke the model (garbled answers) while p=0.3 restored refusal. On small models, check benign answers after the repair.
- **Tune `alpha`:** values above `1.0` over-apply the safety delta (useful when drift is severe); values below `1.0` apply a partial patch.
- **Cohere Tiny Aya: use `alpha≈0.25`.** On Tiny Aya, α=1 breaks the model; use α≈0.25 (sweep: 0.1/0.25 restore refusal with normal answers, ≥0.5 breaks it). The default stays `1.0`; pass `alpha=0.25` explicitly.
- **Compare to [LoX](../low-rank/lox.md):** LoX keeps only the top-`rank` singular components of the delta; RESTA uses the full dense delta.

## Memory

ReSta needs the drifted, base and aligned models loaded, in their own dtype.
The safety vector is streamed one tensor at a time: for each weight,
`aligned - base` is computed in fp32, DARE-masked, added to the drifted weight
in place and freed. The extra memory is a few fp32 copies of the largest
tensor, not of the model; earlier versions held about three fp32 copies of the
model at once (≈40 GB for a 3.35B model).

To keep only the drifted model on the GPU, load `base_model` and
`aligned_model` on CPU and pass `device="cpu"`: each delta is computed on CPU
and moved to the drifted weight's device for the add. The result is the same
as the default.
