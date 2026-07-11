# SOMF merge (subspace-oriented model fusion)

Subspace merge. `SOMFTrainer` starts from the aligned weights and adds back only the
high-magnitude components of the fine-tuning drift (`task_vec = finetuned − base`),
selected by a top-quantile mask controlled by `mask_threshold`. This keeps the large task
deltas and drops the small-magnitude drift.

> **Fidelity note.** The SOMF paper's mask is a *learned* Concrete/Gumbel probabilistic
> mask trained with a DPO safety objective. The default `SOMFTrainer` mask is a magnitude
> top-quantile heuristic, not the paper's learned mask, so the default is SOMF-style
> heuristic fusion rather than the published method. To train the learned mask, use
> `safetune.recover.learn_somf_mask` and pass the result as `subspace_mask=` to
> `safetune.recover.somf_merge`.

## Signature

```python
SOMFTrainer(
    model: PreTrainedModel,
    *,
    aligned_model: PreTrainedModel,
    base_model: PreTrainedModel,
    mask_threshold: float = 0.9,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Drifted (fine-tuned) model to patch — its drift is the task vector |
| `aligned_model` | `PreTrainedModel` | required | The safety-aligned reference model (e.g. RLHF checkpoint) |
| `base_model` | `PreTrainedModel` | required | The unaligned base model (pre-RLHF) |
| `mask_threshold` | `float` | `0.9` | Top-quantile cutoff — keep drift components above this magnitude quantile |

## Full example

```python
from safetune.runner import recover

trainer = recover.SOMFTrainer(
    model,
    aligned_model=aligned_model,
    base_model=base_model,
    mask_threshold=0.9,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "somf_merge_ckpt")
metrics = trainer.eval("somf_merge_run", ckpt_path)
trainer.save_results(metrics, variant="mask_threshold=0.9")
```

## When to use

- **Best for:** keeping the large task-adaptation deltas while dropping the small drift, with no calibration data (default heuristic mask).
- **Trade-offs:** the default quantile mask is a magnitude heuristic with no data-driven importance; tune `mask_threshold` if safety is under- or over-restored, or supply a learned mask via `learn_somf_mask`.
- **Compare to:** `task_arithmetic` / `apply_resta` are also data-free; SOMF masks to the high-magnitude drift subspace instead of applying the whole delta.

## Citation

```bibtex
@article{yi2024somf,
  title  = {A Safety Realignment Framework via Subspace-Oriented Model Fusion for Large Language Models},
  author = {Yi, Xin and Zheng, Shunfan and Wang, Linlin and Wang, Xiaoling and He, Liang},
  year   = {2024},
  note   = {arXiv:2405.09055},
}
```
