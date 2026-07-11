# Constrained SFT — `ConstrainedSFTTrainer`

## ConstrainedSFTTrainer

### Signature

```python
harden.ConstrainedSFTTrainer(
    model=None,
    tokenizer=None,
    *,
    model_id: str = None,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-4,
    bf16: bool = True,
    optimizer: str = "adamw_torch",
    logging_steps: int = 10,
    results_dir: str = None,
    drift_task: str = None,
)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | `None` | Base model to harden |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `model_id` | `str` | `None` | HF path/ID to load the model from when `model` is not passed |
| `epochs` | `int` | `1` | Number of training epochs |
| `batch_size` | `int` | `4` | Per-device train batch size |
| `lr` | `float` | `1e-4` | Learning rate |
| `bf16` | `bool` | `True` | Train in bfloat16 |
| `optimizer` | `str` | `"adamw_torch"` | Optimizer name |
| `logging_steps` | `int` | `10` | Steps between log entries |
| `results_dir` | `str` | `None` | Directory for run outputs |
| `drift_task` | `str` | `None` | Task used to measure post-harden drift |

### ConstrainedSFTConfig fields

| Field | Type | Default | Description |
|---|---|---|---|
| `csft_beta` | `float` | `0.5` | KL penalty scale at position 0; `beta_t = csft_beta * exp(-csft_decay_rate * t)` |
| `csft_decay_rate` | `float` | `0.1` | Exponential decay rate over token positions; higher = constraint concentrated on earlier tokens |

### Full example

```python
from safetune.harden import ConstrainedSFTTrainer, ConstrainedSFTConfig

config = ConstrainedSFTConfig(output_dir="csft_out", csft_beta=0.5, csft_decay_rate=0.1)
trainer = ConstrainedSFTTrainer(
    model=model,
    args=config,
    train_dataset=task_ds,
    reference_model=ref_model,   # frozen aligned model, before fine-tuning
)
trainer.train()
```

The runner wrapper `safetune.runner.harden.ConstrainedSFTTrainer` runs this with
`csft_beta` / `csft_decay_rate` at their config defaults and no reference model,
so it degrades to plain SFT. Use the `safetune.harden` API above to supply
`reference_model` and enable the KL constraint.

### When to use

- **Best for:** a lightweight KL-regularized SFT that penalizes first-token drift from the aligned model.
- **Trade-offs:** Uses a KL-regularized SFT with a position-decaying first-token penalty rather than the paper's bounded-DPO Eq. 3 + step-function β schedule; trains cleanly but cite the implementation, not the paper name.

### Citation

```bibtex
@article{constrainedsft2024,
  title  = {Safety Alignment Should Be Made More Than Just a Few Tokens Deep},
  author = {Qi, et al.},
  year   = {2024},
  note   = {ICLR 2025, arXiv:2406.05946},
}
```
