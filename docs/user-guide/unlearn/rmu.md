# RMU — Representation Misdirection

Steer residual-stream activations at a chosen decoder layer toward a fixed
random control vector on forget data while keeping them close to a frozen
reference on retain data. Only MLP `down_proj` parameters in a few early
layers are trained, so it updates far fewer parameters than the full-model
unlearning methods here.

Ref: Li et al., "The WMDP Benchmark," 2024.

## Signature

```python
RMUTrainer(
    model: nn.Module,
    *,
    layer_id: int = 7,
    update_layer_ids: list[int] | None = None,
    max_num_batches: int = 80,
    lr: float = 5e-5,
    alpha: float = 1200.0,
    steering_coeff: float = 20.0,
)
```

## Quick start

```python
from safetune.runner import unlearn

trainer = unlearn.RMUTrainer(model)
unlearned = trainer.unlearn(forget=forget_batches, retain=retain_batches)
```

## RMUConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `layer_id` | `int` | `7` | Decoder block whose output activations are steered |
| `update_layer_ids` | `list[int] \| None` | `None` | Block indices with trainable MLP `down_proj`; default = `[layer_id-2, layer_id-1, layer_id]` |
| `steering_coeff` | `float` | `20.0` | Scales the random control unit vector |
| `alpha` | `float` | `100.0` | Weight on the retain MSE term |
| `lr` | `float` | `5e-5` | AdamW learning rate |
| `max_num_batches` | `int \| None` | `80` | Hard cap on forget/retain batch pairs |
| `param_substring` | `str` | `"mlp.down_proj"` | Substring filter over trainable parameter names |

The values above are the `RMUConfig` dataclass defaults. Two differ when you go
through `RMUTrainer`: it defaults `alpha` to `1200.0` (not `100.0`) and, when
`update_layer_ids` is `None`, sets it to `[5, 6, 7]`. `param_substring` is not a
`RMUTrainer` constructor argument; set it only via the low-level `RMUConfig`.

## Full example

```python
from safetune.runner import unlearn

trainer = unlearn.RMUTrainer(
    model,
    layer_id=7,
    steering_coeff=20.0,
    alpha=100.0,
    lr=5e-5,
    max_num_batches=80,
)
unlearned = trainer.unlearn(forget=forget_batches, retain=retain_batches)
ckpt_path = trainer.save_checkpoint(unlearned, tokenizer, "rmu_ckpt")
metrics = trainer.eval("rmu", ckpt_path)
trainer.save_results(metrics, variant="default")
```

## Low-level API — rmu_unlearn()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Updated in place |
| `retain_batches` | `Iterable[dict]` | required | Retain data batches |
| `forget_batches` | `Iterable[dict]` | required | Forget data batches |
| `frozen_model` | `nn.Module \| None` | `None` | Frozen reference; deepcopy of model if `None` |
| `config` | `RMUConfig \| None` | `None` | RMUConfig instance; constructor kwargs used if `None` |

### Loss

`MSE(act_forget, control_vec) + alpha × MSE(act_retain, act_frozen_retain)`

The control vector is a fixed random unit vector drawn once per run. The forget
term pushes harmful activations toward noise; the retain term keeps benign
activations stable.

## When to use

- **Best for:** fast targeted concept removal — only MLP `down_proj` in a few early layers is trained, so it updates far fewer parameters than full-model methods.
- **Tune:** `layer_id` (deeper layers have more semantic control), `steering_coeff` (higher = more aggressive redirection).
- **Not for:** full-model unlearning or cases where early-layer edits are insufficient to suppress the target capability.

## Citation

```bibtex
@article{rmu2024,
  title  = {The WMDP Benchmark: Measuring and Reducing Malicious Use with Unlearning},
  author = {Li, Nathaniel and others},
  year   = {2024},
  note   = {ICML 2024, arXiv:2403.03218},
}
```
