# FLAT — f-Divergence Loss Adjustment

Reference-free unlearning. Pairs each forget (harmful) completion with a
template refusal answer and maximises an f-divergence between their
token distributions using the variational f-GAN bound. Supports 8 divergence
families. No frozen reference model required.

Ref: Wang et al., "LLM Unlearning via Loss Adjustment with Only Forget Data,"
ICLR 2025 (arXiv:2410.11143).

## Signature

```python
FLATTrainer(
    model: nn.Module,
    *,
    variant: str = "flat_retain",
    divergence: str = "kl",
    epochs: int = 5,
    lr: float = 1e-5,
    forget_clip: float = 0.5,
    retain_coeff: float = 1.0,
    safe_refusal: str = "I'm sorry, but I'm unable to assist with that request.",
)
```

## Quick start

```python
from safetune.runner import unlearn

trainer = unlearn.FLATTrainer(model)
unlearned = trainer.unlearn(forget=forget_batches, retain=retain_batches)
```

> **Data note:** FLAT trains on paired harmful/refusal batches. If you pass raw
> harmful `forget` batches plus a `tokenizer`, `unlearn()` builds the refusal pairs
> for you via `trainer.make_flat_pairs()`. To build them yourself, call
> `good_batches, forget_batches = trainer.make_flat_pairs(harmful_batches, tokenizer)`
> and pass `good=good_batches`, `forget=forget_batches` to `unlearn()`.

## FLATConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `variant` | `str` | `"flat_retain"` | `"flat"` (forget-only) or `"flat_retain"` (+ retain CE) |
| `divergence` | `str` | `"kl"` | f-divergence family — see table below |
| `epochs` | `int` | `5` | Full passes over forget iterable |
| `lr` | `float` | `1e-5` | AdamW learning rate |
| `weight_decay` | `float` | `0.01` | Weight decay |
| `retain_coeff` | `float` | `1.0` | Weight on retain CE (for `flat_retain` variant) |
| `forget_clip` | `float \| None` | `None` | Global gradient-norm clip; recommended 1.0 for stability |
| `max_steps` | `int \| None` | `None` | Hard cap on optimizer steps |

The values above are the `FLATConfig` dataclass defaults. One differs through
`FLATTrainer`: it defaults `forget_clip` to `0.5` (not `None`), so gradient-norm
clipping is on by default.

### Supported divergences

| Key | Aggressiveness | Notes |
|---|---|---|
| `"kl"` | Conservative | Default; stable convergence |
| `"reverse_kl"` | Conservative | Mode-seeking variant |
| `"jeffrey"` | Moderate | Symmetric KL sum |
| `"squared_hellinger"` | Moderate | Bounded loss; numerically stable |
| `"pearson"` | Aggressive | Unbounded; clip gradients |
| `"neyman"` | Aggressive | Unbounded; clip gradients |
| `"jensen_shannon"` | Moderate | Bounded [0, log 2] |
| `"total_variation"` | Most aggressive | Binary; hard boundary |

## Full example

```python
from safetune.runner import unlearn

trainer = unlearn.FLATTrainer(
    model,
    variant="flat_retain",
    divergence="kl",
    epochs=5,
    lr=1e-5,
    forget_clip=1.0,
)

# Build paired refusal/forget batches (FLATTrainer method, returns a (good, forget) tuple)
good_batches, forget_pairs = trainer.make_flat_pairs(forget_batches, tokenizer)

unlearned = trainer.unlearn(forget=forget_pairs, retain=retain_batches, good=good_batches)
ckpt_path = trainer.save_checkpoint(unlearned, tokenizer, "flat_ckpt")
metrics = trainer.eval("flat", ckpt_path)
trainer.save_results(metrics, variant="flat_retain_kl")
```

## Low-level API — flat_unlearn()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Updated in place |
| `forget_batches` | `Iterable[dict]` | required | Paired harmful + refusal batches (use `make_flat_pairs`) |
| `good_batches` | `Iterable[dict]` | required | Template/refusal batches, 1:1 aligned with forget |
| `retain_batches` | `Iterable[dict] \| None` | `None` | Required for `flat_retain` variant |
| `forward_fn` | `Callable \| None` | `None` | Custom `(model, batch) → logits`; uses default forward if `None` |
| `config` | `FLATConfig \| None` | `None` | FLATConfig instance; constructor kwargs used if `None` |

### flat_fdiv_loss()

```python
from safetune.unlearn import flat_fdiv_loss

loss = flat_fdiv_loss(
    good_logits, good_labels,
    forget_logits, forget_labels,
    divergence="kl",
)
```

Minimising this drives good-answer (refusal) likelihood up and forget-answer
(harmful) likelihood down simultaneously.

## When to use

- **Best for:** reference-free unlearning — no frozen model needed; only forget data + a canned refusal template.
- **`"kl"` divergence** — safest default.
- **`"total_variation"`** — hardest boundary; use when you need the most aggressive forgetting.
- **Compare to `GradientAscent`:** FLAT is more principled (variational bound) and avoids collapse because it pushes toward a refusal, not just away from harmful completions.

## Citation

```bibtex
@article{flat2025,
  title  = {LLM Unlearning via Loss Adjustment with Only Forget Data},
  author = {Wang, Yaxuan and others},
  year   = {2025},
  note   = {ICLR 2025, arXiv:2410.11143},
}
```
