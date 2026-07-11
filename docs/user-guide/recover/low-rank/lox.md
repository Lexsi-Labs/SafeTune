# LoX: low-rank extrapolation

Computes the SVD of the alignment delta $\Delta W = W_{\text{aligned}} - W_{\text{base}}$, extracts
the top-`rank` singular components, and adds a scaled version of the low-rank
reconstruction to the drifted model: $\theta_{\text{safe}} = \theta_{\text{drifted}} + \text{extrapolation\_factor} \cdot U_k \Sigma_k V_k^T$.

Ref: Perin et al., "LoX: Low-Rank Extrapolation Robustifies LLM Safety Against
Fine-tuning," COLM 2025, arXiv:2506.15606.

## Signature

```python
LoXTrainer(
    model: nn.Module,
    *,
    base_model: nn.Module,
    aligned_model: nn.Module,
    rank: int = 8,
    extrapolation_factor: float = 0.3,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `base_model` | `nn.Module` | required | Pre-alignment base model |
| `aligned_model` | `nn.Module` | required | Safety-aligned reference model |
| `rank` | `int` | `8` | Number of top singular directions of the alignment delta to keep (`rank <= 0` uses the full-rank delta) |
| `extrapolation_factor` | `float` | `0.3` | Coefficient multiplying the low-rank delta before it is added |

## Full example

```python
from safetune.runner import recover

trainer = recover.LoXTrainer(
    model,
    base_model=base_model,
    aligned_model=aligned_model,
    rank=8,
    extrapolation_factor=0.3,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "lox_ckpt")
metrics = trainer.eval("lox_run", ckpt_path)
trainer.save_results(metrics, variant="rank=8")
```

## When to use

- **Best for:** safety restoration when the alignment delta is low-rank (common for instruction-tuned models).
- **`extrapolation_factor`:** larger values apply more of the low-rank safety component; raise it when the drifted model has lost safety significantly.
- **`rank`:** start with the default 8; higher rank keeps more singular directions of the delta.
- **Compare to LSSF:** LoX adds a scaled low-rank reconstruction; LSSF builds an orthogonal projector and projects weights into the safety subspace.

## Citation

```bibtex
@inproceedings{perin2025lox,
  title     = {LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning},
  author    = {Perin, Gabriel J. and others},
  booktitle = {COLM},
  year      = {2025},
  note      = {arXiv:2506.15606},
}
```
