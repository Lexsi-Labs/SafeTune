# Pre-FT subspace extrapolation — `LoXHardenTrainer`

## LoXHardenTrainer

### Signature

```python
# Runner API (recommended)
from safetune.runner import harden

trainer = harden.LoXHardenTrainer(
    model,                                   # the ALIGNED model (it is hardened and fine-tuned)
    tokenizer,
    base_model_path="Qwen/Qwen2.5-0.5B",     # the pre-alignment base it was aligned from
    rank=8,
    extrapolation_factor=0.3,
)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | The **aligned** model: it is extrapolated in place (`W_aligned + coef * lowrank(W_aligned - W_base)`) and then fine-tuned |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer for the subsequent SFT pass |
| `base_model_path` | `str` | required | HF path/ID of the pre-alignment base model; raises `ValueError` if missing. `aligned_model_path` is no longer accepted (raises) |
| `rank` | `int` | `8` | Number of top singular directions of `(W_aligned − W_base)` to extrapolate |
| `extrapolation_factor` | `float` | `0.3` | Extrapolation strength along the safety subspace |

### Full example

```python
from safetune.runner import harden

trainer = harden.LoXHardenTrainer(
    model,                                   # the aligned model, e.g. Qwen/Qwen2.5-0.5B-Instruct
    tokenizer,
    base_model_path="Qwen/Qwen2.5-0.5B",
    rank=8,
    extrapolation_factor=0.3,
)
# .train() applies the LoX pre-FT extrapolation, then fine-tunes on your task data.
out_path = trainer.train(task_dataset)
```

### When to use

- **Best for:** extrapolating the aligned model along the top-k singular directions of (W_aligned − W_base) *before* fine-tuning, so the safety subspace is reinforced.
- **Trade-offs:** Training-free pre-FT step; adds a preprocessing pass before fine-tuning begins.

### Citation

```bibtex
@article{loxharden2025,
  title  = {LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning},
  author = {Perin, et al.},
  year   = {2025},
  note   = {COLM 2025, arXiv:2506.15606},
}
```
