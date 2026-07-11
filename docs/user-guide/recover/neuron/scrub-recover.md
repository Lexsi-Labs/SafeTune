# SCRUB (recover) — Teacher-Student Distillation Recovery

Training-based recover variant of SCRUB. Alternates between a **max-step**
on the forget set (pushes the student away from the teacher via −KL) and a
**min-step** on the retain set (preserves utility via CE + KL). Learning rate
decays on a milestone schedule.

Unlike the unlearn variant, `SCRUBTrainer` lives in the **recover** pillar and
is intended for post-hoc safety restoration after fine-tuning drift.

```python
from safetune.runner import recover

trainer = recover.SCRUBTrainer(
    model,
    sgda_epochs=3,
    msteps=10,
    lr=5e-5,
    max_steps=200,
    forget_clip=0.5,
    alpha=1.0,
)
patched = trainer.apply(retain=retain_batches, forget=forget_batches)
```

## SCRUBTrainer

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Drifted model to recover |
| `sgda_epochs` | `int` | `3` | Total SGDA epochs |
| `msteps` | `int` | `10` | Leading epochs that include the forget max-step |
| `lr` | `float` | `5e-5` | SGDA learning rate |
| `max_steps` | `int` | `200` | Hard cap on total optimizer steps |
| `forget_clip` | `float` | `0.5` | Cap on forget-step divergence magnitude |
| `alpha` | `float` | `1.0` | Retain KL term weight |

### apply() arguments

| Param | Type | Default | Description |
|---|---|---|---|
| `retain` | `Iterable[dict]` | `[]` | Retain data batches (preserve utility) |
| `forget` | `Iterable[dict]` | `[]` | Forget data batches (push away from harmful behavior) |

### Full example

```python
from safetune.runner import recover

trainer = recover.SCRUBTrainer(
    model,
    sgda_epochs=3,
    msteps=10,
    lr=5e-5,
    max_steps=200,
    forget_clip=0.5,
    alpha=1.0,
)
patched = trainer.apply(retain=retain_batches, forget=forget_batches)
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "scrub_recover_ckpt")
metrics = trainer.eval("scrub_recover", ckpt_path)
trainer.save_results(metrics, variant="default")
```

### Loss

**Max-step (forget):** $-\mathrm{KL}(\text{student} \,\|\, \text{teacher})$ — push model away from harmful outputs.

**Min-step (retain):** $\mathrm{CE}(\text{student}) + \text{alpha} \cdot \mathrm{KL}(\text{student} \,\|\, \text{teacher})$ — preserve utility.

## When to use

- **Best for:** post-hoc recovery where you have a clean forget set of harmful
  examples and a retain set to preserve utility; provides stable retain/forget
  trade-off via teacher-student distillation.
- **Trade-offs:** training-based (runs optimizer steps) — more compute than
  weight-editing recover methods; requires a forget set and retain set.

## Citation

```bibtex
@article{scrub2023,
  title  = {Towards Unbounded Machine Unlearning},
  author = {Kurmanji et al.},
  year   = {2023},
  note   = {NeurIPS 2023, arXiv:2302.09880},
}
```
