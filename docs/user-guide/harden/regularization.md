# Weight-space regularization — `AsFTTrainer`, `BoosterTrainer`, `SaLoRATrainer`

Methods that regularize the fine-tuning update in weight space so it stays
close to the aligned model's safety-relevant parameters.

## AsFTTrainer

### Signature

```python
AsFTTrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    reg_lambda: float = 1.0,
    aligned_model_path: str | None = None,
    **kwargs,
)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | `None` | Base model (LoRA path recommended) |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `reg_lambda` | `float` | `1.0` | Lagrangian weight on the orthogonal-component penalty (LoRA path) |
| `aligned_model_path` | `str` | `None` | Path to the aligned reference model; defaults to the tokenizer's `name_or_path` |

### AsFTConfig fields

| Field | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `str` | required | Checkpoint output path |
| `reg_lambda` | `float` | `1.0` | Lagrangian weight on the orthogonal-component penalty (LoRA path) |
| `hard_constraint` | `bool` | `False` | If True, fully suppresses the orthogonal gradient component instead of penalizing it |

### Full example

```python
from safetune.runner import harden

trainer = harden.AsFTTrainer(model, tokenizer, reg_lambda=1.0)
trainer.train(task_dataset)
```

### When to use

- **Best for:** LoRA fine-tunes where you want to constrain the adapter update away from safety-critical directions.
- **Trade-offs:** LoRA-only path; works on the adapter delta, not full weights.

### Citation

```bibtex
@article{asft2025,
  title  = {AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin},
  author = {Yang, et al.},
  year   = {2025},
  note   = {NeurIPS 2025, arXiv:2506.08473},
}
```

---

## SaLoRA

The **safety subspace** here is a low-rank set of weight-space directions
derived from the aligned model (via `compute_safety_subspace`, either from
the `aligned − base` delta or from safety-prompt activations) — not the
same thing as the safety neurons/circuits Interpret locates; it's specific
to this projection-based defense. `SaLoRATrainer` below wraps these same
two functions into a `Trainer`-style API — see its "When to use" for which
form to reach for.

### Signature

```python
from safetune.harden import compute_safety_subspace, project_lora_step

# Derive the per-parameter safety subspace from the aligned model
# (returns a dict of projection bases keyed by parameter name).
safety_subspace = compute_safety_subspace(
    aligned,             # nn.Module — aligned reference model
    base=None,           # optional nn.Module — pre-alignment base
    rank=8,
    safety_inputs=None,  # optional iterable of tokenized safety batches
)

# Install the safety projection onto `model`; returns the number of
# modules projected.
n_projected = project_lora_step(model, safety_subspace)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `aligned` | `nn.Module` | required | Aligned reference model the safety subspace is derived from |
| `base` | `nn.Module \| None` | `None` | Optional pre-alignment base; when given, the subspace is built from the `aligned − base` delta |
| `rank` | `int` | `8` | Number of singular directions kept per parameter |
| `safety_inputs` | `Iterable[dict] \| None` | `None` | Optional tokenized safety batches for a data-driven (activation) subspace |

### Full example

```python
from safetune.harden import compute_safety_subspace, project_lora_step

safety_inputs = [tokenizer(p, return_tensors="pt") for p in safety_prompts]
safety_subspace = compute_safety_subspace(aligned_model, safety_inputs=safety_inputs, rank=8)

n_projected = project_lora_step(model, safety_subspace)
```

### When to use

- **Best for:** LoRA fine-tunes where you want the task adapter to stay in the safety-orthogonal subspace.
- **Trade-offs:** requires a pre-computed safety module from activation statistics.

### Citation

```bibtex
@article{salora2025,
  title  = {SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation},
  author = {Li, et al.},
  year   = {2025},
  note   = {ICLR 2025, arXiv:2501.01765},
}
```

---

## BoosterTrainer

### Signature

```python
BoosterTrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    perturb_scale: float = 0.01,
    **kwargs,
)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | `None` | Model to train |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `perturb_scale` | `float` | `0.01` | Step size `alpha` of the simulated one-step harmful SGD attack used for the finite-difference regularizer |

### Full example

```python
from safetune.runner import harden

trainer = harden.BoosterTrainer(model, tokenizer, perturb_scale=0.01)
trainer.train(task_ds, out_dir="./booster-model")
```

`harmful_batches` are sampled automatically from contamination sets if not supplied.

### When to use

- **Best for:** alignment-stage defense that appends a finite-difference harmful-gradient regularizer to the SFT objective — attenuating directions along which a harmful attacker would quickly reduce the harmful loss.
- **Trade-offs:** requires two backward passes through the harmful set per outer step (clean + perturbed). Larger `perturb_scale` increases robustness but can hurt task accuracy.

### Citation

```bibtex
@article{booster2025,
  title  = {Booster: Tackling Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Perturbation},
  author = {Huang, et al.},
  year   = {2025},
  note   = {ICLR 2025 Oral, arXiv:2409.01586},
}
```

---

## SaLoRATrainer

### Signature

```python
SaLoRATrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    rank: int = 16,
    safety_rank: int = 16,
    strength: float = 1.0,
    task_init: bool = True,
    n_iter: int = 7,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    **kwargs,
)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | `None` | Model to train |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `rank` | `int` | `16` | LoRA adapter rank |
| `safety_rank` | `int` | `16` | Number of top singular vectors defining the safety subspace |
| `strength` | `float` | `1.0` | Projection strength — how strongly the adapter update is kept orthogonal to the safety subspace |
| `task_init` | `bool` | `True` | Initialize LoRA `A` with the task direction for faster convergence |
| `n_iter` | `int` | `7` | Power-iteration steps for the incremental GPU SVD |
| `lora_alpha` | `int` | `2 * rank` | LoRA scaling factor |
| `lora_dropout` | `float` | `0.05` | LoRA dropout rate |
| `target_modules` | `list[str]` | attn + MLP projections | Module name substrings to apply LoRA to |

### Full example

```python
from safetune.runner import harden

trainer = harden.SaLoRATrainer(model, tokenizer,
                                rank=16, safety_rank=16, strength=1.0)
trainer.train(task_ds, out_dir="./salora-model")
```

`safety_dataset` is loaded automatically from calibration data if not supplied.

### When to use

- **Best for:** LoRA fine-tunes where you want the task adapter update to stay orthogonal to the safety-critical representation subspace (extracted via incremental GPU SVD on safety calibration data).
- **Trade-offs:** `safety_rank` controls the dimensionality of the protected subspace — higher values preserve more safety at the cost of available task capacity in the adapter.
- **vs. `SaLoRA` above:** same projection method; this is the `Trainer`-style wrapper (auto-loads `safety_dataset`, drives the full `train()` loop). Use the plain functions above if you're managing the training loop yourself.

### Citation

```bibtex
@article{salora2025,
  title  = {SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation},
  author = {Li, et al.},
  year   = {2025},
  note   = {ICLR 2025, arXiv:2501.01765},
}
```
