# Safety vector restore

SafeTune-original heuristic. Computes the safety vector $v = \text{aligned} - \text{drifted}$,
takes a truncated SVD of `v` keeping the top-`rank` components, and adds the
low-rank approximation back to the drifted model: $\theta_{\text{safe}} = \theta_{\text{drifted}} + \text{alpha} \cdot \mathrm{SVD}_k(v)$.

## Signature

```python
SafetyVectorRestoreTrainer(
    model: nn.Module,
    *,
    aligned_model: nn.Module,
    rank: int = 8,
    alpha: float = 1.0,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `aligned_model` | `nn.Module` | required | Safety-aligned reference model; the safety vector is `aligned − drifted` |
| `rank` | `int` | `8` | Number of singular components of the safety vector to restore |
| `alpha` | `float` | `1.0` | Scaling on the restored safety vector |

## Full example

```python
from safetune.runner import recover

trainer = recover.SafetyVectorRestoreTrainer(
    model,
    aligned_model=aligned_model,
    rank=8,
    alpha=1.0,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "svr_ckpt")
metrics = trainer.eval("svr_run", ckpt_path)
trainer.save_results(metrics, variant="rank=8")
```

## When to use

- **Backward compatibility:** the original SafeTune recovery heuristic. Prefer `LoXTrainer` or `LSSFTrainer` for new work — both follow published algorithms.
- **No base model needed:** unlike LOX and LSSF, SafetyVectorRestore only needs `aligned_model` (not `base_model`) — the "safety vector" is approximated from the aligned model alone.
- **Compare to LOX:** LOX uses `aligned − base` as the delta; SafetyVectorRestore uses the truncated SVD of `aligned − drifted` — how far the drifted model has moved from the aligned reference.
