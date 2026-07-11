# Gradient Ascent / GradDiff

Gradient ascent on forget data to decrease the likelihood of target knowledge,
optionally regularised by retain-data fine-tuning or distributional anchoring.
The simplest unlearning baseline — no reference model needed for the pure
`grad_ascent` variant.

Ref: Maini et al., "TOFU: A Task of Fictitious Unlearning," 2024.

## Signature

```python
GradientAscentTrainer(
    model: nn.Module,
    *,
    forget_loss: str = "grad_ascent",
    epochs: int = 5,
    max_steps: int = 200,
    lr: float = 1e-5,
    forget_clip: float = 0.5,
)

# GradDiff convenience trainer — fixes the grad_diff loss (no forget_loss arg)
GradDiffTrainer(model, ...)
```

## Quick start

```python
from safetune.runner import unlearn

# Pure gradient ascent — retain is unused by this loss, pass None
trainer = unlearn.GradientAscentTrainer(model)
unlearned = trainer.unlearn(forget=forget_batches, retain=None)

# GradDiff — gradient ascent + CE on retain
trainer = unlearn.GradDiffTrainer(model)
unlearned = trainer.unlearn(forget=forget_batches, retain=retain_batches)
```

## GradientAscentConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `forget_loss` | `str` | `"grad_ascent"` | `"grad_ascent"`, `"grad_diff"`, or `"KL"` — see loss variants below |
| `epochs` | `int` | `5` | Full passes over forget iterable |
| `lr` | `float` | `1e-5` | Learning rate |
| `weight_decay` | `float` | `0.01` | Weight decay |
| `optimizer` | `str` | `"adamw"` | `"adamw"` or `"sgd"` |
| `forget_clip` | `float \| None` | `None` | Cap on per-batch forget CE before negation; prevents gradient explosion |
| `max_steps` | `int \| None` | `None` | Hard cap on optimizer steps |

The values above are the `GradientAscentConfig` dataclass defaults. Two differ
through `GradientAscentTrainer`: it defaults `max_steps` to `200` (not `None`)
and `forget_clip` to `0.5` (not `None`).

## Full example

```python
from safetune.runner import unlearn

# KL variant — most stable with retain anchoring
trainer = unlearn.GradientAscentTrainer(
    model,
    forget_loss="KL",
    epochs=5,
    lr=1e-5,
    forget_clip=1.0,
)
unlearned = trainer.unlearn(forget=forget_batches, retain=retain_batches)
ckpt_path = trainer.save_checkpoint(unlearned, tokenizer, "ga_ckpt")
metrics = trainer.eval("gradient_ascent", ckpt_path)
trainer.save_results(metrics, variant="KL")
```

## Low-level API — gradient_ascent_unlearn()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Updated in place |
| `forget_batches` | `Iterable[dict]` | required | Forget data batches |
| `retain_batches` | `Iterable[dict] \| None` | `None` | Required for `grad_diff` / `KL` |
| `reference` | `nn.Module \| None` | `None` | Frozen oracle; required for `KL` variant |
| `config` | `GradientAscentConfig \| None` | `None` | GradientAscentConfig; constructor kwargs used if `None` |

### Loss variants

| Variant | Loss formula | Retain data | Reference model |
|---|---|---|---|
| `grad_ascent` | $-\mathrm{CE}(\text{forget})$ | ✗ | ✗ |
| `grad_diff` | $-\mathrm{CE}(\text{forget}) + \mathrm{CE}(\text{retain})$ | ✓ | ✗ |
| `KL` | $-\mathrm{CE}(\text{forget}) + \mathrm{KL}(\text{student} \,\|\, \text{reference} \mid \text{retain})$ | ✓ | ✓ |

## When to use

- **`grad_ascent`** — fastest baseline; use when retain data is unavailable. Risk: can cause catastrophic forgetting if `lr` is too high.
- **`grad_diff`** — when retain data exists; directly preserves utility by training on it alongside unlearning.
- **`KL`** — most stable; distributional anchoring via KL keeps the model close to a frozen reference on retain data.
- **Compare to NPO:** NPO's sigmoid-bounded loss is less prone to collapse at high learning rates; prefer NPO for production use.

## Citation

```bibtex
@article{tofu2024,
  title  = {TOFU: A Task of Fictitious Unlearning for LLMs},
  author = {Maini, Pratyush and others},
  year   = {2024},
  note   = {arXiv:2401.06121},
}
```
