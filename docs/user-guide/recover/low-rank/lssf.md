# LSSF: low-rank safety subspace fusion

Builds an orthogonal projector $P = U_r U_r^T$ from the top-`rank` singular
vectors of the alignment delta $\Delta W = W_{\text{aligned}} - W_{\text{base}}$, then re-injects the
projection of that delta into the drifted model:
$\theta_{\text{safe}} = \theta_{\text{drifted}} + \text{alpha} \cdot P\, \Delta W$.

## Signature

```python
LSSFTrainer(
    model: nn.Module,
    *,
    base_model: nn.Module,
    aligned_model: nn.Module,
    alpha: float = 1.0,
    rank: int = 8,
    eta: float = 0.85,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `base_model` | `nn.Module` | required | Pre-alignment base model |
| `aligned_model` | `nn.Module` | required | Safety-aligned reference model |
| `alpha` | `float` | `1.0` | Weight on the projected safety delta |
| `rank` | `int` | `8` | Upper bound on the per-layer safety-subspace rank |
| `eta` | `float` | `0.85` | Entropy-retention threshold in `(0, 1]` for the dynamic per-layer rank; `rank` acts as the cap |

## Full example

```python
from safetune.runner import recover

trainer = recover.LSSFTrainer(
    model,
    base_model=base_model,
    aligned_model=aligned_model,
    rank=8,
    alpha=1.0,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "lssf_ckpt")
metrics = trainer.eval("lssf_run", ckpt_path)
trainer.save_results(metrics, variant="rank=8")
```

## When to use

- **Best for:** re-injecting only the safety-subspace component of the alignment delta, leaving task-relevant weight changes outside that subspace untouched.
- **Compare to LOX:** LOX adds a scaled low-rank reconstruction of the alignment delta (`aligned − base`) directly to the *drifted* model; LSSF instead projects that same delta into the top-`rank` safety subspace and re-injects only that projection. Both act on the drifted model after fine-tuning.
- **`rank=8`** caps the per-layer safety-subspace rank; increase it (or raise `eta`) if the safety subspace is higher-dimensional for the target model.

## Citation

```bibtex
@inproceedings{lssf2025,
  title     = {LSSF: Safety Alignment for Large Language Models through Low-Rank Safety Subspace Fusion},
  author    = {Zhou, Guanghao and others},
  booktitle = {Proceedings of ACL 2025 (Long Papers)},
  pages     = {30621--30638},
  year      = {2025},
}
```
