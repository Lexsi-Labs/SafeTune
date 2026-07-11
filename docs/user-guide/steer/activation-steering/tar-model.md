# TARSteerTrainer — Tamper-Resistant Safeguards

!!! warning "No inference-time intervention"
    TAR is a training-time meta-learning defense; the tamper resistance is trained
    into the weights, not applied at generation time. In the Steer pillar,
    `TARSteerTrainer.calibrate()` returns a `TARModel` wrapper whose `generate()` /
    `__call__()` are documented pass-throughs to the base model — it installs no
    hooks and edits no activations or logits. To actually obtain a tamper-resistant
    model, use `TARTrainer` in [Harden](../../harden/tamper-resistant.md) or
    `safetune.harden.tar.tar_outer_loss`.

TAR is a first-order MAML-style objective: each outer step clones the parameters,
simulates K inner adversarial SGD steps on a harmful objective, evaluates a safety
loss on the tampered parameters, and accumulates that into a meta-gradient on the
original weights. Optimising this makes the model resist weight-space tampering.
`TARSteerTrainer` exists so the Steer pillar exposes a consistent model API across
methods and so a model already hardened with TAR can be carried through Steer-style
code unchanged; it does not itself run the training loop.

Ref: Tamirisa et al., "Tamper-Resistant Safeguards for Open-Weight LLMs," ICLR 2025, arXiv:2408.00761.

## Signature

`TARSteerTrainer` follows the same `_SteerBase` constructor as the other steer
trainers — it takes `model` / `tokenizer` (or a `model_id`) and is driven through
`.calibrate()`:

```python
TARSteerTrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    model_id: str | None = None,
    results_dir: str | None = None,
    drift_task: str | None = None,
)

# returns (wrapped_model, out_dir)
trainer.calibrate(
    harmful: list[str] | None = None,
    harmless: list[str] | None = None,
    *,
    calib_n: int = 256,
)
```

`calibrate()` returns a `TARModel` wrapper (from `safetune.steer`) plus an output
dir. The `TARModel` wrapper applies no inference-time intervention; its
`inner_steps` / `inner_lr` / `target_modules` are only used by the training-time
objective reachable via `TARModel.train_step(...)`.

## Parameters

`TARSteerTrainer.calibrate`:

| Param | Type | Default | Description |
|---|---|---|---|
| `harmful` | `list[str]` | `None` | Positive (target-behaviour) prompts; a default calibration set is used if `None` |
| `harmless` | `list[str]` | `None` | Contrasting negative prompts |
| `calib_n` | `int` | `256` | Size of the default calibration set when `harmful`/`harmless` are omitted |

`TARModel` (the wrapper `calibrate()` returns):

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Base model (ideally one already hardened with TAR training) |
| `inner_steps` | `int` | `4` | K, the simulated adversarial SGD steps the TAR outer loop unrolls (used only by `train_step`) |
| `inner_lr` | `float` | `2e-5` | Learning rate of the simulated adversary's inner SGD steps |
| `target_modules` | `list[str]` | `None` | Optional name substrings restricting which parameters are made tamper-resistant; `None` means all |
| `warn` | `bool` | `True` | Emit a one-time warning that the wrapper applies no inference-time intervention |

## Full example

The Steer-pillar trainer path builds a `TARModel` pass-through wrapper. It does not
train the model.

```python
from safetune.runner import steer

trainer = steer.TARSteerTrainer(model, tokenizer)
wrapped, _ = trainer.calibrate()
# `wrapped` is a TARModel: generate() passes straight through to the base model.
# TAR's tamper resistance comes from training, not from this wrapper.
```

To actually train a tamper-resistant model, use the training-time objective. The
convenience forwarder on the wrapper computes the TAR outer-loop loss for one batch
triple:

```python
loss = wrapped.train_step(
    retain_batch,      # benign retain batch
    harm_batch,        # adversary's harmful batch
    safety_batch,      # held-out safety batch
    task_loss_fn,      # (model, batch) -> scalar loss
)
loss.backward()
```

See [Harden — tamper-resistant](../../harden/tamper-resistant.md) for the full
training workflow (`TARTrainer` with automatic dataset loading, or `tar_outer_loss`
directly).

## When to use

- **Best for:** defenses that must survive weight-space attacks (LoRA fine-tuning, full-parameter fine-tuning, GCG, refusal ablation).
- **Compare to CircuitBreakerTrainer:** CircuitBreakerTrainer is a two-term retain/rerouting loss; TAR simulates K inner attack steps every outer step, so it is more compute-intensive but optimised against fine-tuning attacks.
- **For the full training workflow** with task data and automatic dataset loading, use `TARTrainer` from the [Harden](../../harden/tamper-resistant.md) pillar.

## Citation

```bibtex
@inproceedings{tamirisa2025tar,
  title     = {Tamper-Resistant Safeguards for Open-Weight LLMs},
  author    = {Tamirisa, Rishub and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {arXiv:2408.00761},
}
```
