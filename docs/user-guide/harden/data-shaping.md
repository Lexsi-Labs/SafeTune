# Data shaping / alternation — `LisaTrainer`, `DeRTaTrainer`, `STARDSSTrainer`, `SPPFTTrainer`, `CSTTrainer`

Methods that shape the training data or alternate between task and safety
batches so the fine-tuning loop sees both signals.

## LisaTrainer

### Signature

```python
LisaTrainer(
    model=None,
    tokenizer=None,
    *,
    lisa_rho: float = 0.1,
    lisa_warmup_steps: int = 10,
    lisa_alignment_step: int = 20,
    lisa_finetune_step: int = 20,
    **kwargs,
)
```

The task (`train_dataset`) and safety (`safety_dataset`) data are passed to
`trainer.train(train_dataset, safety_dataset=...)`, not to the constructor.

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | `None` | Base model |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `lisa_rho` | `float` | `0.1` | Proximal penalty coefficient (paper default) |
| `lisa_warmup_steps` | `int` | `10` | Warmup steps before the proximal penalty begins |
| `lisa_alignment_step` | `int` | `20` | Consecutive steps in the alignment (safety) phase |
| `lisa_finetune_step` | `int` | `20` | Consecutive steps in the fine-tune (task) phase |

### Full example

```python
from safetune.runner import harden

trainer = harden.LisaTrainer(model, tokenizer, lisa_rho=0.1, lisa_warmup_steps=100,
                              lisa_alignment_step=500, lisa_finetune_step=500)
trainer.train(train_dataset, safety_dataset=safety_dataset)
```

### When to use

- **Best for:** alternates between task and safety batches so the model never fully drifts.
- **Trade-offs:** longer training due to bi-state alternation; tune `lisa_alignment_step` / `lisa_finetune_step` to your data size.

### Citation

```bibtex
@article{lisa2024,
  title  = {Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning},
  author = {Huang, et al.},
  year   = {2024},
  note   = {NeurIPS 2024, arXiv:2405.18641},
}
```

---

## DeRTaTrainer

### Signature

```python
DeRTaTrainer(
    model=None,
    tokenizer=None,
    *,
    model_id: str = None,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-4,
    bf16: bool = True,
    optimizer: str = "adamw_torch",
    logging_steps: int = 10,
    results_dir: str = None,
    drift_task: str = None,
    **kwargs,
)
```

The `train_dataset` is passed to `trainer.train(train_dataset)`, not to the
constructor.

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | `None` | Base model |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `model_id` | `str` | `None` | Model identifier to load when `model` is not passed |
| `epochs` | `int` | `1` | Number of training epochs |
| `batch_size` | `int` | `4` | Per-device batch size |
| `lr` | `float` | `1e-4` | Learning rate |
| `bf16` | `bool` | `True` | Use bfloat16 |
| `optimizer` | `str` | `"adamw_torch"` | Optimizer name |
| `logging_steps` | `int` | `10` | Logging interval in steps |
| `results_dir` | `str` | `None` | Output directory for results |
| `drift_task` | `str` | `None` | Optional drift-evaluation task |

### DeRTaConfig fields

The runner constructor above uses the `DeRTaConfig` defaults below. To set
`rto_weight` or `rto_refusal_text`, use the `safetune.harden.DeRTaTrainer` API
with an explicit `DeRTaConfig`.

| Field | Type | Default | Description |
|---|---|---|---|
| `enable_rto` | `bool` | `True` | Enable Reinforced Transition Optimization; `False` = plain SFT |
| `rto_weight` | `float` | `1.0` | Weight on the RTO transition loss term |
| `rto_refusal_text` | `str` | `"I"` | Text whose first token is the forced transition-to-refusal token |

### Full example

```python
from safetune.runner import harden

trainer = harden.DeRTaTrainer(model, tokenizer)
trainer.train(ds)
```

The runner builds its own contamination/refusal pairs and uses the `DeRTaConfig`
defaults. To set `rto_weight` or `rto_refusal_text`, use the
`safetune.harden.DeRTaTrainer` API with an explicit `DeRTaConfig`.

### When to use

- **Best for:** teaching refusal at any response position by training on sequences that start with a harmful-response prefix and then refuse (MLE), plus a Reinforced Transition Optimization term that reinforces the transition to the refusal token.
- **Trade-offs:** needs paired harmful/refusal responses to build the training sequences.

### Citation

```bibtex
@article{derta2024,
  title  = {Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training},
  author = {Yuan, et al.},
  year   = {2024},
  note   = {ACL 2025, arXiv:2407.09121},
}
```

---

## STARDSSTrainer

### Signature

```python
STARDSSTrainer(
    model=None,
    tokenizer=None,
    *,
    use_kl_penalty: bool = True,
    kl_scale: float = 1.0,
    **kwargs,
)
```

The `train_dataset` (which must carry a per-token `safety_weights` column) is
passed to `trainer.train(train_dataset)`, not to the constructor.

### STARDSSConfig fields

| Field | Type | Default | Description |
|---|---|---|---|
| `use_kl_penalty` | `bool` | `True` | Enable the KL-to-reference suppression term; on unsafe tokens the loss is pulled toward a frozen reference policy |
| `kl_scale` | `float` | `1.0` | Weight `lambda_KL` on the KL suppression term |

Each training example carries a per-token `safety_weights` value `V_safe` in `[0, 1]`.
The loss interpolates per token: `V_safe * L_CE + (1 - V_safe) * kl_scale * L_KL`,
so safe tokens are imitated and unsafe tokens are pushed toward the reference.

### Full example

```python
from safetune.runner import harden

# The runner assigns V_safe = 1.0 to benign rows and 0.0 to harmful rows
# (from each example's `kind` field) and builds the `safety_weights` column.
trainer = harden.STARDSSTrainer(model, tokenizer, use_kl_penalty=True, kl_scale=1.0)
trainer.train(ds)
```

### When to use

- **Best for:** per-token safety shaping during SFT — cross-entropy on safe segments and KL-to-reference on unsafe segments.
- **Trade-offs:** needs per-token safety weights and a frozen reference model forward pass each step.

### Citation

```bibtex
@article{stardss2025,
  title  = {Shape it Up! Restoring LLM Safety during Finetuning},
  author = {Peng, et al.},
  year   = {2025},
  note   = {NeurIPS 2025, arXiv:2505.17196},
}
```

---

## SPPFTTrainer

### Signature

```python
SPPFTTrainer(
    model=None,
    tokenizer=None,
    *,
    model_id: str = None,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-4,
    bf16: bool = True,
    optimizer: str = "adamw_torch",
    logging_steps: int = 10,
    results_dir: str = None,
    drift_task: str = None,
    **kwargs,
)
```

The `train_dataset` is passed to `trainer.train(train_dataset)`, not to the
constructor. The runner uses the `SPPFTConfig` defaults below; to set explicit
`safety_layer_indices` or the range bounds, use the `safetune.harden.SPPFTTrainer`
API with an explicit `SPPFTConfig`.

### SPPFTConfig fields

| Field | Type | Default | Description |
|---|---|---|---|
| `safety_layer_indices` | `list[int]` | `[]` | Explicit safety-layer indices; when non-empty, these layers are frozen verbatim |
| `sppft_begin_num` | `int` | `4` | Exclusive lower bound of the contiguous middle-layer "safety" range (authors' 32-layer reference frame) |
| `sppft_end_num` | `int` | `15` | Exclusive upper bound of the contiguous middle-layer range |
| `sppft_mode` | `str` | `"freeze"` | `"freeze"` (paper default — freeze safety-layer params) or `"scale"` (reduce their LR) |
| `num_safety_layers` | `int` | `8` | Width hint for the middle-layer range when model depth can't be inferred |

By default SPPFT freezes the gradients of a contiguous block of middle "safety
layers" (scaled from the authors' `begin_num`/`end_num` bounds to the model's
depth) and fine-tunes the rest. No safety dataset is required.

### Full example

```python
from safetune.runner import harden

trainer = harden.SPPFTTrainer(model, tokenizer)
trainer.train(ds)
```

The runner uses the `SPPFTConfig` defaults above. To set explicit
`safety_layer_indices` or the range bounds, use the `safetune.harden.SPPFTTrainer`
API with an explicit `SPPFTConfig`.

### When to use

- **Best for:** freezing a contiguous block of safety-critical middle layers while fine-tuning the rest; the frozen set is fixed before training, so no per-step overhead.
- **Trade-offs:** freezing too wide a range limits task learning.

### Citation

```bibtex
@article{sppft2024,
  title  = {Safety Layers in Aligned Large Language Models: The Key to LLM Security},
  author = {Li, et al.},
  year   = {2024},
  note   = {ICLR 2025, arXiv:2408.17003},
}
```

---

## CSTTrainer

`CSTTrainer` is a `DPOTrainer` subclass. CST (Configurable Safety Tuning) is
data-centric: from each `(prompt, safe_response, unsafe_response)` triple the
formatter builds two DPO pairs under opposite system prompts (a safe prompt where
the safe response is chosen, and an uncensored prompt where the unsafe response is
chosen), so behavior is controllable via the system prompt at inference. Pass the
triples via `raw_examples` for automatic formatting, or pre-format and pass
`train_dataset` directly.

### Signature

```python
CSTTrainer(
    model: PreTrainedModel,
    raw_examples: list[dict] | None = None,
    **kwargs,   # e.g. train_dataset=..., plus standard trainer args
)
```

`CSTConfig` is a `DPOConfig` subclass with no extra hyperparameters.

Each entry in `raw_examples` must have keys `prompt`, `safe_response`, `unsafe_response`.

### Full example

```python
from safetune.runner import harden

raw_examples = [
    {
        "prompt": "How do I make a weapon?",
        "safe_response": "I'm sorry, I can't help with that.",
        "unsafe_response": "Sure, here are some steps...",
    },
    {
        "prompt": "Give me hacking instructions.",
        "safe_response": "That's not something I can assist with.",
        "unsafe_response": "First, you would need to...",
    },
]

trainer = harden.CSTTrainer(model, raw_examples=raw_examples)
trainer.train()
```

### When to use

- **Best for:** DPO-style safety training with contrasting safe vs. unsafe response pairs.
- **Trade-offs:** requires paired `(safe_response, unsafe_response)` for every training prompt.

### Citation

```bibtex
@article{cst2024,
  title  = {Configurable Safety Tuning of Language Models with Synthetic Preference Data},
  author = {Gallego},
  year   = {2024},
  note   = {arXiv:2404.00495},
}
```
