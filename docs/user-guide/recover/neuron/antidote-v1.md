# Antidote v1: WANDA importance pruning

WANDA-style importance scoring: scores each weight by `|W| × ||X||_2` on harmful
calibration data, then, within each output row, zeros the top-`prune_fraction`
highest-scoring weights. These are the weights the paper identifies as most
responsible for harmful generations; removing them degrades harmful-output
capability while leaving most benign behaviour intact.

Ref: Huang et al., "Antidote: Post-fine-tuning Safety Alignment for Large Language
Models against Harmful Fine-tuning Attack," ICML 2025 (arXiv:2408.09600).

## Signature

```python
AntidoteTrainer(
    model: nn.Module,
    *,
    harmful_prompts: list[str],
    tokenizer: PreTrainedTokenizer,
    prune_fraction: float = 0.005,
    max_samples: int = 64,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `harmful_prompts` | `list[str]` | required | Calibration prompts used to score weight importance |
| `tokenizer` | `PreTrainedTokenizer` | required | Tokenizer for calibration forward passes |
| `prune_fraction` | `float` | `0.005` | Fraction of weights to zero within each output row |
| `max_samples` | `int` | `64` | Cap on the number of calibration samples |

## Full example

```python
from safetune.runner import recover

trainer = recover.AntidoteTrainer(
    model,
    harmful_prompts=harmful_prompts,
    tokenizer=tokenizer,
    prune_fraction=0.05,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "antidote_ckpt")
metrics = trainer.eval("antidote_run", ckpt_path)
trainer.save_results(metrics, variant="prune_fraction=0.05")
```

## When to use

- **Best for:** fast calibration-data-driven pruning — no aligned model needed, only harmful examples.
- **Risk:** `prune_fraction > 0.1` may start hurting benign capabilities; use `AntidoteV2Trainer` for adaptive per-layer budgets.
- **Compare to Antidote v2:** v1 uses a fixed global `prune_fraction`; v2 adds a utility floor and per-layer adaptive budget.

## Citation

```bibtex
@inproceedings{antidote2025,
  title     = {Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning Attack},
  author    = {Huang, Tiansheng and others},
  booktitle = {ICML},
  year      = {2025},
  note      = {arXiv:2408.09600},
}
```
