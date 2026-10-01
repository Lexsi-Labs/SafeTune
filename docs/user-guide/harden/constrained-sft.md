# Constrained SFT — `ConstrainedSFTTrainer`

## ConstrainedSFTTrainer

### Signature

```python
harden.ConstrainedSFTTrainer(
    model=None,
    tokenizer=None,
    *,
    model_id: str = None,
    reference_model_path: str = None,
    use_reference: bool = None,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-4,
    bf16: bool | None = None,
    fp16: bool | None = None,
    wandb: bool = False,
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
| `reference_model_path` | `str` | `None` | HF path/ID of the frozen aligned reference for the KL constraint; `None` loads the model being fine-tuned (the tokenizer's `name_or_path`) |
| `use_reference` | `bool` | `None` | `None` is `True` (KL constraint on) unless `safetune.configure(legacy_constrained_sft=True)`; `False` trains plain SFT |
| `epochs` | `int` | `1` | Number of training epochs |
| `batch_size` | `int` | `4` | Per-device train batch size |
| `lr` | `float` | `1e-4` | Learning rate |
| `bf16` | `bool \| None` | `None` | Train in bfloat16. `None`: from the runtime dtype (`safetune.configure(dtype=...)`; bf16 where supported) |
| `fp16` | `bool \| None` | `None` | Train in float16. `None`: from the runtime dtype |
| `wandb` | `bool` | `False` | Log to Weights & Biases |
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
from safetune.harden import ConstrainedSFTHFTrainer, ConstrainedSFTConfig

config = ConstrainedSFTConfig(output_dir="csft_out", csft_beta=0.5, csft_decay_rate=0.1)
trainer = ConstrainedSFTHFTrainer(
    model=model,
    args=config,
    train_dataset=task_ds,
    reference_model=ref_model,   # frozen aligned model, before fine-tuning
)
trainer.train()
```

`ConstrainedSFTHFTrainer` is the `transformers.Trainer` subclass behind
`harden.ConstrainedSFTTrainer`. The high-level trainer (also used by the CLI)
accepts `csft_beta` / `csft_decay_rate` as keyword arguments, loads the frozen
reference from `reference_model_path` and passes it to
`ConstrainedSFTHFTrainer`, so the KL constraint is on. Before, it passed no
reference and trained plain SFT; `use_reference=False` or
`safetune.configure(legacy_constrained_sft=True)` gives that behaviour.

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
