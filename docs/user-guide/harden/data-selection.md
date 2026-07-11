# Data selection — `SEALTrainer`

## SEALTrainer

### Signature

```python
harden.SEALTrainer(
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
    **kwargs,
)
```

The safety/task datasets are passed to `trainer.train(train_dataset, safety_dataset=...)`,
not to the constructor (see the [Full example](#full-example)).

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | `None` | Base model (or load via `model_id`) |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `model_id` | `str` | `None` | Hub id / path to load the model + tokenizer from |
| `epochs` | `int` | `1` | Number of training epochs |
| `batch_size` | `int` | `4` | Per-device train batch size |
| `lr` | `float` | `1e-4` | Learning rate |
| `bf16` | `bool` | `True` | Train in bfloat16 |
| `optimizer` | `str` | `"adamw_torch"` | Optimizer name |
| `logging_steps` | `int` | `10` | Steps between log entries |
| `results_dir` | `str` | `None` | Output directory for checkpoints/results |
| `drift_task` | `str` | `None` | Optional drift-evaluation task |

### SEALConfig fields

| Field | Type | Default | Description |
|---|---|---|---|
| `seal_temperature` | `float` | `1.0` | Softmax temperature for importance weights; lower = more aggressive up-weighting |
| `seal_rescore_every` | `int` | `10` | Steps between gradient-conflict re-scoring passes |
| `seal_top_k_ratio` | `float` | `1.0` | Fraction of examples to up-weight (1.0 = weight all) |

### Full example

```python
from safetune.runner import harden

trainer = harden.SEALTrainer(model, tokenizer)
trainer.train(task_ds, safety_dataset=safety_ds)
```

The runner wrapper uses the `SEALConfig` defaults above. To set `seal_temperature`
or `seal_rescore_every`, use the `safetune.harden.SEALTrainer` API with an explicit
`SEALConfig`. `safety_dataset` is built automatically if not passed.

### When to use

- **Best for:** up-weighting SFT examples whose gradient conflicts most with the alignment gradient (tractable SEAL approximation).
- **Trade-offs:** Requires computing per-example gradient conflict scores; adds an upstream data-scoring step.

### Citation

```bibtex
@article{seal2024,
  title  = {SEAL: Safety-Enhanced Aligned LLM Fine-Tuning via Bilevel Data Selection},
  author = {Shen, et al.},
  year   = {2024},
  note   = {arXiv:2410.07471},
}
```
