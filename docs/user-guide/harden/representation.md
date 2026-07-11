# Representation perturbation — `VaccineTrainer`, `TVaccineTrainer`, `SAPTrainer`, `SurgeryTrainer`

Methods that perturb or shape hidden-state representations during fine-tuning
so the model's safety-related internal structure survives the update.

## vaccine_loss

### Signature

```python
from safetune.harden import vaccine_loss, VaccineConfig

config = VaccineConfig(rho=1e-3)
loss = vaccine_loss(model, train_batch, task_loss_fn, config=config)
loss.backward()
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model being fine-tuned |
| `batch` | `dict` | required | Your task training batch |
| `task_loss_fn` | `callable` | required | Loss function: `lambda m, b: m(**b).loss` |
| `config` | `VaccineConfig` | `VaccineConfig()` | Perturbation config (`rho` controls magnitude) |

### Full example

```python
from safetune.harden import vaccine_loss, VaccineConfig

config = VaccineConfig(rho=1e-3)
task_loss_fn = lambda m, b: m(**b).loss

for batch in dataloader:  # your own tokenized batches, with "labels" set
    optimizer.zero_grad()
    loss = vaccine_loss(model, batch, task_loss_fn, config=config)
    loss.backward()
    optimizer.step()
```

### When to use

- **Best for:** immunizing the model against harmful fine-tuning by perturbing representations during SFT.
- **Trade-offs:** adds a perturbation pass per step; use with a standard SFT trainer that calls `vaccine_loss` as its loss.

### Citation

```bibtex
@article{vaccine2024,
  title  = {Vaccine: Perturbation-Aware Alignment for Large Language Models against Harmful Fine-Tuning},
  year   = {2024},
  note   = {arXiv:2402.01109},
}
```

---

## VaccineTrainer

### Signature

```python
VaccineTrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    rho: float = 2.0,
    **kwargs,
)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | `None` | Model to train |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `rho` | `float` | `2.0` | SAM perturbation radius applied to attention-layer output hidden states |

### Full example

```python
from safetune.runner import harden

trainer = harden.VaccineTrainer(model, tokenizer, rho=2.0)
trainer.train(task_ds, out_dir="./vaccine-model")
```

### When to use

- **Best for:** immunizing the model during alignment-stage SFT by perturbing attention-layer hidden states with a globally normalized SAM step before computing the task loss, making representations robust to later harmful fine-tuning.
- **Trade-offs:** adds a forward + autograd pass per step; larger `rho` increases safety robustness at a small accuracy cost.

### Citation

```bibtex
@article{vaccine2024,
  title  = {Vaccine: Perturbation-Aware Alignment for Large Language Models against Harmful Fine-Tuning},
  author = {Huang, et al.},
  year   = {2024},
  note   = {NeurIPS 2024, arXiv:2402.01109},
}
```

---

## tvaccine_loss (TVaccineTrainer)

### Signature

```python
# Runner-style API (safetune.runner.harden)
TVaccineTrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    rho: float = 2.0,
    top_k_ratio: float = 0.5,
    **kwargs,
)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | `None` | Model being fine-tuned |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `rho` | `float` | `2.0` | SAM perturbation radius |
| `top_k_ratio` | `float` | `0.5` | Fraction of attention layers to perturb (ranked by gradient norm) |

### Full example

```python
from safetune.runner.harden import TVaccineTrainer

trainer = TVaccineTrainer(
    model=model, tokenizer=tokenizer,
    rho=2.0, top_k_ratio=0.5,
)
trainer.train(train_dataset=task_ds, out_dir="./tvaccine-model")
```

### When to use

- **Best for:** memory-efficient Vaccine — only perturbs the top-k safety-critical layers instead of all.
- **Trade-offs:** Perturbs only the top-k safety-critical layers — more memory-efficient than full Vaccine, with a similar safety–capability trade-off.

### Citation

```bibtex
@article{tvaccine2024,
  title  = {T-Vaccine: memory-efficient, layer-selective perturbation-aware alignment},
  author = {Liu, et al.},
  year   = {2024},
  note   = {arXiv:2410.09760},
}
```

---

## SAPTrainer

### Signature

```python
SAPTrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    grad_rate: float = 0.1,
    v_update_step: float = 0.05,
    contrastive_temperature: float = 1.0,
    **kwargs,
)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | `None` | Base model |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `grad_rate` | `float` | `0.1` | Scale of harmful perturbation $\Delta W_{\text{harmful}}$; actual $\varepsilon = \text{grad\_rate} \cdot \|W\|$ |
| `v_update_step` | `float` | `0.05` | Inner probe step size $\beta$ (bilevel optimization) |
| `contrastive_temperature` | `float` | `1.0` | Temperature of the contrastive safety loss $L_{\text{safe}}$ |

The contrastive safety batches (`safety_dataloader`) are passed to `trainer.train(...)`, not the constructor — see the Full example below. They carry the keys `input_ids`, `attention_mask`, `chosen_labels`, `rejected_labels`.

### Full example

```python
from safetune.runner import harden

trainer = harden.SAPTrainer(model, tokenizer, grad_rate=0.1, v_update_step=0.05,
                             contrastive_temperature=1.0)
trainer.train(task_ds, safety_dataloader=safety_loader)
```

### When to use

- **Best for:** bilevel training with a hidden-state probe that simulates a harmful weight perturbation `ΔW_harmful` each step and trains the model to keep the safe-useful gap under it.
- **Trade-offs:** requires contrastive safety batches with `chosen_labels` / `rejected_labels`.

### Citation

```bibtex
@article{sap2025,
  title  = {Secure LLM Fine-Tuning via Safety-Aware Probing (SAP)},
  author = {Wu, et al.},
  year   = {2025},
  note   = {arXiv:2505.16737},
}
```

---

## SurgeryTrainer

### Signature

```python
SurgeryTrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    sink_lambda: float = 0.01,
    **kwargs,
)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | `None` | Base model |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `sink_lambda` | `float` | `0.01` | Attention-sink divergence regularizer weight |

The separate harmful batch (`harmful_dataset`) is passed to `trainer.train(...)`, not the constructor — see the Full example below. If absent, the trainer falls back to using the task batch as X_m (documented shortcut).

### Full example

```python
from safetune.runner import harden

trainer = harden.SurgeryTrainer(model, tokenizer, sink_lambda=0.01)
trainer.train(ds, harmful_dataset=harmful_ds)
```

Adds an attention-sink divergence regularizer: penalizes head-level attention
on position 0 that drifts above the pre-finetune reference.

### When to use

- **Best for:** preventing attention-sink drift that erodes refusal behavior.
- **Trade-offs:** pass `harmful_dataset` for the faithful Algorithm 1 path; without it, falls back to using the task batch.

### Citation

```bibtex
@article{surgery2026,
  title  = {Surgery: Mitigating Harmful Fine-Tuning for Large Language Models via Attention Sink},
  year   = {2026},
  note   = {arXiv:2602.05228},
}
```
