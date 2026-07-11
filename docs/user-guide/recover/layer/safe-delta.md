# Safe Delta: OBS-style weight edit

Entry-wise weight edit inspired by Optimal Brain Surgeon (OBS). Starts from the
aligned weights and selectively re-introduces the fine-tuning delta
`W_sft − W_aligned` entry by entry: it keeps the delta entries that buy utility
and reverts the entries that erode safety, under a global budget set by
`strength`.

Ref: Lu et al., "Safe Delta: Consistently Preserving Safety when Fine-Tuning
LLMs on Diverse Datasets," ICML 2025, arXiv:2505.12038.

## Signature

```python
SafeDeltaTrainer(
    model: nn.Module,
    *,
    aligned_model: nn.Module,
    strength: float = 0.1,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `aligned_model` | `nn.Module` | required | Safety-aligned reference model |
| `strength` | `float` | `0.1` | Safety budget `s`. Larger values keep more of the fine-tuning delta (more utility, weaker safety recovery); `0` reverts entirely to the aligned weights |

## Full example

```python
from safetune.runner import recover

trainer = recover.SafeDeltaTrainer(
    model,
    aligned_model=aligned_model,
    strength=0.1,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "safe_delta_ckpt")
metrics = trainer.eval("safe_delta_run", ckpt_path)
trainer.save_results(metrics, variant="strength=0.1")
```

## When to use

- **Best for:** minimal-footprint recovery — the safety budget bounds how much of the fine-tuning delta is kept.
- **Tune `strength`:** lower values recover more safety by reverting more entries toward the aligned weights; `0` reverts entirely.
- **Compare to RESTA:** RESTA adds the full dense delta; SafeDelta applies a budget-constrained entry-wise edit.
