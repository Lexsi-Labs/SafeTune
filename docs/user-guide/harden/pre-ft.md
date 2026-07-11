# Pre-FT subspace extrapolation — `LoXHardenTrainer`

## LoXHardenTrainer

### Signature

```python
# Runner API (recommended)
from safetune.runner import harden

trainer = harden.LoXHardenTrainer(
    model,                                   # base model to harden
    tokenizer,
    aligned_model_path="Qwen/Qwen2.5-0.5B-Instruct",  # aligned reference
    rank=8,
    extrapolation_factor=0.3,
)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | The base model to harden (its weights are the LoX base; modified in-place) |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer for the subsequent SFT pass |
| `aligned_model_path` | `str` | `None` | HF path/ID of the safety-aligned reference model (falls back to the tokenizer's model id) |
| `rank` | `int` | `8` | Number of top singular directions of `(W_aligned − W_base)` to extrapolate |
| `extrapolation_factor` | `float` | `0.3` | Extrapolation strength along the safety subspace |

### Full example

```python
from safetune.runner import harden

trainer = harden.LoXHardenTrainer(
    model,
    tokenizer,
    aligned_model_path="Qwen/Qwen2.5-0.5B-Instruct",
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
  title  = {LoX Harden: Pre-FT Safety-Subspace Extrapolation},
  author = {Perin, et al.},
  year   = {2025},
  note   = {COLM 2025, arXiv:2506.15606},
}
```
