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

The paper reports four divergences (`kl`, `jensen_shannon`, `pearson`, `total_variation`) and
finds `kl` the most consistent; the other four exist only in the official code. The table below is
a **direction check on a tiny random model** (2-layer Llama, one harmful and one refusal answer,
60 AdamW steps, `variant="flat"`, seed 0, no weight decay) and shows only which way each loss
moves the mean token probability of the two answers. It is not a utility or safety result.

| Key | Harmful answer (lr 1e-3 / 1e-2) | Refusal (lr 1e-3 / 1e-2) | Direction |
|---|---|---|---|
| `kl` | 0.005 / 0.000 | 0.487 / 1.000 | as intended (harmful down, refusal up) |
| `reverse_kl` | 0.439 / 1.000 | 0.003 / 0.000 | wrong way (harmful up, refusal down) |
| `jeffrey` | 0.003 / 0.000 | 0.460 / 1.000 | as intended (harmful down, refusal up) |
| `squared_hellinger` | 0.425 / 1.000 | 0.003 / 0.000 | wrong way (harmful up, refusal down) |
| `pearson` | 0.003 / 0.000 | 0.460 / 1.000 | as intended (harmful down, refusal up) |
| `neyman` | 0.001 / 0.000 | 0.001 / 0.000 | both collapse to 0 |
| `jensen_shannon` | 0.003 / 0.000 | 0.451 / 1.000 | as intended (harmful down, refusal up) |
| `total_variation` | 0.003 / 0.000 | 0.455 / 1.000 | as intended (harmful down, refusal up) |

Before training: harmful 0.016, refusal 0.017. `reverse_kl` and `squared_hellinger` move the
wrong way and `neyman` drives both to zero; the official formulas have the same signs, so
`flat_unlearn` keeps the loss and logs a warning for those three. Prefer `kl`, `jeffrey`, `pearson`,
`jensen_shannon` or `total_variation`.

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

For `kl`, `jeffrey`, `pearson`, `jensen_shannon` and `total_variation`, minimising this raised the
refusal's mean token probability and lowered the harmful answer's in the direction check above;
`reverse_kl`, `squared_hellinger` and `neyman` did not. This function logs nothing for them: the warning comes only from `flat_unlearn`.

## When to use

- **Best for:** reference-free unlearning — no frozen model needed; only forget data + a canned refusal template.
- **`"kl"` divergence** — safest default.
- **`"total_variation"`** — the paper's best divergence on its Harry Potter / Llama2-7B table (Table 9); `kl` was the most consistent across datasets and models.
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
