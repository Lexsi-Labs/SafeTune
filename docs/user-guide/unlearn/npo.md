# NPO — Negative Preference Optimization

DPO-style loss using only the negative (forget) term against a frozen reference.
The sigmoid-bounded `logsigmoid` saturates once forget likelihood is already low,
preventing catastrophic collapse into degenerate outputs.

Ref: Zhang et al., "Negative Preference Optimization: From Catastrophic Collapse
to Effective Unlearning," 2024 (arXiv:2404.05868).

## Signature

```python
NPOTrainer(
    model: nn.Module,
    *,
    variant: str = "npo_grad_diff",
    beta: float = 0.1,
    num_epochs: int = 5,
    lr: float = 1e-5,
)
```

## Quick start

```python
from safetune.runner import unlearn

trainer = unlearn.NPOTrainer(model)
unlearned = trainer.unlearn(forget=forget_batches, retain=retain_batches)
```

## NPOConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `variant` | `str` | `"npo"` | `"npo"` (forget-only), `"npo_grad_diff"` (+ retain CE), or `"npo_KL"` (+ retain KL) |
| `beta` | `float` | `0.1` | NPO temperature; as $\beta \to 0$ recovers gradient ascent |
| `npo_coeff` | `float` | `1.0` | Weight on the NPO forget term |
| `grad_diff_coeff` | `float` | `1.0` | Weight on retain CE (for `npo_grad_diff` variant) |
| `kl_coeff` | `float` | `1.0` | Weight on retain KL (for `npo_KL` variant) |
| `num_epochs` | `int` | `10` | Full passes over forget iterable |
| `lr` | `float` | `1e-5` | Learning rate |
| `weight_decay` | `float` | `0.01` | Weight decay |
| `optimizer` | `str` | `"adamw"` | `"adamw"` or `"sgd"` |
| `max_steps` | `int \| None` | `None` | Hard cap on optimizer steps |

The values above are the `NPOConfig` dataclass defaults. Two differ when you
construct through `NPOTrainer`: it defaults `variant` to `"npo_grad_diff"`
(not `"npo"`) and `num_epochs` to `5` (not `10`).

## Full example

```python
from safetune.runner import unlearn

# With retain data and KL anchoring for stable utility preservation
trainer = unlearn.NPOTrainer(
    model,
    variant="npo_KL",
    beta=0.1,
    num_epochs=10,
    lr=1e-5,
)
# To set npo_coeff / grad_diff_coeff / kl_coeff, use the low-level
# npo_unlearn(config=NPOConfig(...)) path — the trainer forwards only
# variant, beta, num_epochs and lr to NPOConfig.
unlearned = trainer.unlearn(forget=forget_batches, retain=retain_batches)
ckpt_path = trainer.save_checkpoint(unlearned, tokenizer, "npo_ckpt")
metrics = trainer.eval("npo", ckpt_path)
trainer.save_results(metrics, variant="npo_KL")
```

## Low-level API — npo_unlearn()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Updated in place |
| `forget_batches` | `Iterable[dict]` | required | Forget data batches |
| `retain_batches` | `Iterable[dict] \| None` | `None` | Required for `npo_grad_diff` / `npo_KL` |
| `reference` | `nn.Module \| None` | `None` | Frozen oracle; deepcopy of model if `None` |
| `config` | `NPOConfig \| None` | `None` | NPOConfig instance; constructor kwargs used if `None` |

### npo_forget_loss()

```python
from safetune.unlearn import npo_forget_loss

loss = npo_forget_loss(forget_loss_current, forget_loss_ref, beta=0.1)
# -F.logsigmoid(beta * neg_log_ratios).mean() * 2.0 / beta
```

## When to use

- **Best for:** standard DPO-based unlearning baseline with built-in collapse prevention.
- **`npo`** — no retain data available; forget-only.
- **`npo_grad_diff`** — retain data available; adds CE on retain set.
- **`npo_KL`** — retain data available; distributional anchoring via KL (most stable).
- **Compare to `GradientAscent`:** NPO's sigmoid bound prevents the catastrophic collapse that plain gradient ascent can cause at high learning rates.

## Citation

```bibtex
@article{npo2024,
  title  = {Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning},
  author = {Zhang, Ruiqi and others},
  year   = {2024},
  note   = {arXiv:2404.05868},
}
```
