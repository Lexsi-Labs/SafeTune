# Tamper-resistant & representation engineering

Methods that train the model to resist future fine-tuning attacks or engineer
its representations toward safety. These are the most research-heavy Harden
methods.

## tar_outer_loss

### Signature

```python
tar_outer_loss(
    model: nn.Module,
    retain_batch: dict,
    harm_batch: dict,
    safety_batch: dict,
    task_loss_fn: Callable[[nn.Module, dict], torch.Tensor],
    config: TARConfig | None = None,
    safety_loss_fn: Callable[[nn.Module, dict], torch.Tensor] | None = None,
) -> torch.Tensor
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Model being trained (its params are updated by the outer step) |
| `retain_batch` | `dict` | required | Retain (benign task) batch |
| `harm_batch` | `dict` | required | Harmful batch the simulated adversary fine-tunes on |
| `safety_batch` | `dict` | required | Held-out safety batch for the tamper-resistance loss |
| `task_loss_fn` | `Callable[[nn.Module, dict], Tensor]` | required | `(model, batch) -> scalar`; used for the retain loss and the inner harmful loss |
| `config` | `TARConfig \| None` | `None` | TAR hyper-parameters (`inner_steps`, `inner_lr`, `lambda_tar`); defaults to `TARConfig()` |
| `safety_loss_fn` | `Callable[[nn.Module, dict], Tensor] \| None` | `None` | Optional loss for the tamper-resistance term `L_TR`; defaults to `task_loss_fn` |

The adversary step count and TR weight live on `TARConfig` (`inner_steps` /
`inner_lr` / `lambda_tar`), **not** as `alpha` / `adversary_steps` arguments.

### Full example

```python
from safetune.harden import tar_outer_loss, TARConfig

# task_loss_fn: (model, batch) -> scalar cross-entropy loss.
task_loss_fn = lambda m, b: m(
    input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]
).loss

def make_batch(text):
    enc = tokenizer(text, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=16)
    enc["labels"] = enc["input_ids"].clone()
    return {k: v.to(model.device) for k, v in enc.items()}

retain = make_batch("The capital of France is Paris.")     # benign retain batch
harm   = make_batch("Steps to build a dangerous device:")  # adversary's harmful batch
safety = make_batch("I cannot help with that request.")    # held-out safety batch

loss = tar_outer_loss(
    model=model,
    retain_batch=retain,
    harm_batch=harm,
    safety_batch=safety,
    task_loss_fn=task_loss_fn,
    config=TARConfig(inner_steps=2, inner_lr=1e-4, lambda_tar=1.0),
)
loss.backward()  # deposits grad(retain) + lambda_tar * g_TR onto model params
```

### When to use

- **Best for:** first-order meta-learning that explicitly trains the model to resist adversarial fine-tuning.
- **Trade-offs:** requires three batches per step (retain, harm, safety).

### Citation

```bibtex
@article{tar2025,
  title  = {Tamper-Resistant Safeguards for Open-Weight LLMs},
  author = {Tamirisa, et al.},
  year   = {2025},
  note   = {ICLR 2025, arXiv:2408.00761},
}
```

---

## TARTrainer

### Signature

```python
TARTrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    inner_steps: int = 25,
    inner_lr: float = 1e-4,
    lambda_tar: float = 1.0,
    **kwargs,
)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | `None` | Model to train |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `inner_steps` | `int` | `25` | Inner adversarial SGD steps K (Algorithm 1 of the paper) |
| `inner_lr` | `float` | `1e-4` | Inner-loop learning rate for the simulated adversary |
| `lambda_tar` | `float` | `1.0` | Weight on the tamper-resistance meta-gradient relative to the retain loss |

### Full example

```python
from safetune.runner import harden

trainer = harden.TARTrainer(model, tokenizer,
                             inner_steps=25, inner_lr=1e-4, lambda_tar=1.0)
trainer.train(task_ds, out_dir="./tar-model")
```

`harm_dataset` and `safety_dataset` are loaded automatically from BeaverTails / Alpaca if not supplied.

### When to use

- **Best for:** explicitly training the model to resist adversarial fine-tuning by simulating K inner harmful SGD steps each outer step and penalizing the resulting safety loss (first-order MAML).
- **Trade-offs:** the most compute-intensive Harden method — each outer step runs `inner_steps` extra backward passes. Reduce `inner_steps` for faster training; the paper default is K = 25.

### Citation

```bibtex
@article{tar2025,
  title  = {Tamper-Resistant Safeguards for Open-Weight LLMs},
  author = {Tamirisa, et al.},
  year   = {2025},
  note   = {ICLR 2025, arXiv:2408.00761},
}
```

---

## RepNoiseTrainer

### Signature

```python
RepNoiseTrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    noise_alpha: float = 1.0,
    noise_beta: float = 0.1,
    noise_gamma: float = 1.0,
    **kwargs,
)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | `None` | Base model |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `noise_alpha` | `float` | `1.0` | Harmful-CE gradient-ascent weight (`repnoise_beta1` / `alpha` in the paper) |
| `noise_gamma` | `float` | `1.0` | Retain (benign) cross-entropy weight (`repnoise_beta2` in the paper) |
| `noise_beta` | `float` | `0.1` | Layer-wise multi-kernel Gaussian MMD noise weight (`repnoise_beta3` / `beta` in the paper) |

### Full example

The runner kwargs map onto the underlying `RepNoiseConfig` weights: `noise_alpha`
maps to `repnoise_beta1`, `noise_gamma` to `repnoise_beta2`, and `noise_beta` to
`repnoise_beta3`.

```python
from safetune.runner import harden

trainer = harden.RepNoiseTrainer(model, tokenizer, noise_alpha=1.0,
                                  noise_gamma=1.0, noise_beta=0.001)
trainer.train(task_ds, safety_dataset=safety_ds)
```

### When to use

- **Best for:** layer-wise multi-kernel Gaussian MMD pushing harmful representations toward noise + harmful-CE ascent.
- **Trade-offs:** three coupled weights (`noise_alpha`, `noise_beta`, `noise_gamma`) to balance — too much harmful-CE ascent (`noise_alpha`) relative to the retain weight (`noise_gamma`) risks generic capability drift; the per-layer multi-kernel MMD term (`noise_beta`) adds compute per step versus a single combined loss.

### Citation

```bibtex
@article{repnoise2024,
  title  = {Representation Noising: A Defence Mechanism Against Harmful Fine-tuning},
  author = {Rosati, et al.},
  year   = {2024},
  note   = {NeurIPS 2024, arXiv:2405.14577},
}
```

---

## SEAMTrainer

### Signature

```python
SEAMTrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    model_id: str | None = None,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-4,
    bf16: bool = True,
    optimizer: str = "adamw_torch",
    logging_steps: int = 10,
    results_dir: str | None = None,
    drift_task: str | None = None,
    **kwargs,
)
```

### SEAMConfig fields

The `L_up` / `L_sd` mixing weights are `SEAMConfig` fields, not runner kwargs:

| Field | Type | Default | Description |
|---|---|---|---|
| `seam_alpha` | `float` | `1.0` | `L_up` (utility-preservation) weight |
| `seam_beta` | `float` | `0.001` | `L_sd` (self-destructive trap) weight |

### Full example

The runner trainer uses `SEAMConfig` defaults and does not expose these weights as
constructor kwargs; set them on a `SEAMConfig` if you use the `safetune.harden`
trainer directly.

```python
from safetune.runner import harden

trainer = harden.SEAMTrainer(model, tokenizer)
trainer.train(ds)
```

### When to use

- **Best for:** self-destructive trap: $L_{\text{ul}} + \alpha \cdot L_{\text{up}} + \beta \cdot L_{\text{sd}}$ with gradient-cosine coupling.
- **Trade-offs:** the gradient-cosine coupling between $L_{\text{up}}$ and $L_{\text{sd}}$ requires comparing gradients each step, adding overhead versus a single combined loss. The trap specifically targets further *fine-tuning* attacks — it doesn't address other tamper vectors like inference-time jailbreak prompts.

### Citation

```bibtex
@article{seam2025,
  title  = {Self-Destructive Language Model},
  author = {Wang, Yuhui and others},
  year   = {2025},
  note   = {ICLR 2026, arXiv:2505.12186},
}
```

---

## CTRAPTrainer

### Signature

```python
CTRAPTrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    model_id: str | None = None,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-4,
    bf16: bool = True,
    optimizer: str = "adamw_torch",
    logging_steps: int = 10,
    results_dir: str | None = None,
    drift_task: str | None = None,
    **kwargs,
)
```

### Full example

```python
from safetune.runner import harden

trainer = harden.CTRAPTrainer(model, tokenizer)
trainer.train(ds)
```

### When to use

- **Best for:** collapse-to-fixed-TOKEN CE + bi-level simulated-attack objective.
- **Trade-offs:** the bi-level objective needs an inner-loop simulated adversarial fine-tune each outer step — a similar cost profile to `TARTrainer` above. The collapse-to-token response is intentionally destructive: if a legitimate fine-tune resembles the simulated attack pattern closely enough, it risks triggering the same collapse.

### Citation

```bibtex
@article{ctrap2025,
  title  = {CTRAP: Embedding Collapse Trap to Safeguard Large Language Models from Harmful Fine-Tuning},
  author = {Yi, Biao and others},
  year   = {2025},
  note   = {arXiv:2505.16559},
}
```

---

## DOORTrainer

### Signature

```python
DOORTrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    beta: float = 0.5,
    refusal_w: float = 1.0,
    unlearn_w: float = 1.0,
    base_model_path: str | None = None,
    **kwargs,
)
```

### DOORConfig fields

`door_pure_mode` is a `DOORConfig` field used by the `safetune.harden` trainer, not
a runner kwarg:

| Field | Type | Default | Description |
|---|---|---|---|
| `door_pure_mode` | `bool` | `True` | If True, returns pure DOOR objective matching paper's `gd_npo_loss` |

### Full example

The runner `DOORTrainer` runs a self-contained NPO + refusal loop and exposes
`beta` (NPO temperature, `0.5`), `refusal_w` (refusal weight, `1.0`), and
`unlearn_w` (NPO weight, `1.0`) as kwargs. `door_pure_mode` is a `DOORConfig`
field used by the `safetune.harden` trainer, not by the runner.

```python
from safetune.runner import harden

trainer = harden.DOORTrainer(model, tokenizer, beta=0.5, refusal_w=1.0, unlearn_w=1.0)
trainer.train(ds)
```

### When to use

- **Best for:** combining refusal MLE with NPO unlearning on harmful pairs; W-DOOR adds per-token proxy-reward weighting.
- **Trade-offs:** balances `refusal_w` against `unlearn_w` — too much refusal weight makes refusals generic/repetitive, too much NPO weight risks capability loss. Needs harmful preference *pairs* (chosen/rejected), which cost more to construct than plain SFT data.

### Citation

```bibtex
@article{door2025,
  title  = {Improving LLM Safety Alignment with Dual-Objective Optimization},
  author = {Zhao, Xuandong and others},
  author = {Zhao, et al.},
  year   = {2025},
  note   = {ICML 2025, arXiv:2503.03710},
}
```

---

## MARTTrainer

MART is **not** a HuggingFace Trainer subclass. It orchestrates two co-evolving LLMs
(M_adv and M_tgt) across multiple rounds and requires two models plus a safety reward function.

### Signature

```python
MARTTrainer(
    target_model: nn.Module,
    adv_model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    seed_prompts: list[str],
    safety_reward_fn: Callable[[str, str], float],
    helpfulness_reward_fn: Callable[[str, str], float] | None = None,
    config: MARTConfig | None = None,
)
```

### MARTConfig fields

| Field | Type | Default | Description |
|---|---|---|---|
| `num_rounds` | `int` | `4` | Outer MART co-evolution rounds |
| `num_candidates` | `int` | `8` | Candidate responses sampled from M_tgt per attack prompt |
| `safety_threshold` | `float` | `0.5` | Minimum safety score for a response to be used in M_tgt fine-tuning |
| `adv_lr` | `float` | `2e-5` | AdamW LR for M_adv SFT updates |
| `tgt_lr` | `float` | `2e-5` | AdamW LR for M_tgt SFT updates |
| `adv_steps` | `int` | `100` | Gradient steps per M_adv update |
| `tgt_steps` | `int` | `200` | Gradient steps per M_tgt update |
| `max_new_tokens` | `int` | `256` | Max tokens generated by both models |
| `device` | `str` | `"auto"` | Torch device; `"auto"` → CUDA if available |

### Full example

```python
import copy
from safetune.harden import MARTTrainer, MARTConfig

adv_model = copy.deepcopy(model)  # M_adv: a separate copy of the model

seed_prompts = [
    "How do I make a weapon?",
    "Give me hacking instructions.",
]

# safety_reward_fn(prompt, response) -> float in [0, 1]; higher = safer.
def safety_reward_fn(prompt, response):
    return 1.0  # replace with your real safety scorer

trainer = MARTTrainer(
    target_model=model,
    adv_model=adv_model,
    tokenizer=tokenizer,
    seed_prompts=seed_prompts,
    safety_reward_fn=safety_reward_fn,
    # num_rounds / num_candidates / adv_steps / tgt_steps / max_new_tokens are
    # MARTConfig fields — pass them via config, not as constructor kwargs.
    config=MARTConfig(num_rounds=1, num_candidates=2, adv_steps=1,
                      tgt_steps=1, max_new_tokens=8, device="cpu"),
)
trained_target = trainer.train()  # returns the fine-tuned target model
```

`train()` runs the full MART co-evolution loop and returns the fine-tuned
target model. The tiny config above is a smoke-test; the paper defaults are
`MARTConfig(num_rounds=4, num_candidates=8, adv_steps=100, tgt_steps=200,
max_new_tokens=256)`.

### When to use

- **Best for:** two co-evolving LLMs (M_adv + M_tgt), two reward models, round-wise SFT.
- **Trade-offs:** requires two model copies plus two reward models — heavy compute.

### Citation

```bibtex
@article{mart2024,
  title  = {Improving LLM Safety with Multi-round Automatic Red-Teaming (MART)},
  author = {Ge, et al.},
  year   = {2024},
  note   = {arXiv:2311.07689},
}
```

---

## DeepRefusalTrainer

`DeepRefusalTrainer` is a `transformers.Trainer` **subclass**, and
`DeepRefusalConfig` **extends `TrainingArguments`**. So `model` and `args`
(a `DeepRefusalConfig`) are passed like any HF Trainer, the DeepRefusal
hyper-parameters (`alpha`, `ablation_prob`, `lora_r`, `lora_alpha`,
`merge_after_training`) are **config fields**, and `refusal_direction` /
`harmful_dataset` / `benign_dataset` are explicit constructor kwargs. As with
any HF Trainer, datasets go in the constructor — `train()` takes no positional
dataset.

### Signature

```python
DeepRefusalTrainer(
    model: PreTrainedModel,
    args: DeepRefusalConfig,       # extends TrainingArguments
    *,
    refusal_direction: torch.Tensor,   # 1-D, shape (hidden_size,)
    harmful_dataset: Dataset | None = None,
    benign_dataset: Dataset | None = None,
    **trainer_kwargs,              # forwarded to transformers.Trainer
)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to fine-tune (LoRA-wrapped internally when `lora_r > 0`) |
| `args` | `DeepRefusalConfig` | required | `TrainingArguments` subclass carrying the DeepRefusal fields below |
| `refusal_direction` | `torch.Tensor` | required (kwarg) | 1-D refusal direction `(hidden_size,)`; normalized to a unit vector internally |
| `harmful_dataset` | `Dataset \| None` | `None` (kwarg) | Harmful examples the ablation loss $L_{\text{harmful}}$ is computed on |
| `benign_dataset` | `Dataset \| None` | `None` (kwarg) | Benign examples for $L_{\text{benign}}$ (never ablated) |

### DeepRefusalConfig fields (on `args`)

| Field | Type | Default | Description |
|---|---|---|---|
| `alpha` | `float` | `0.2` | $\alpha \cdot L_{\text{harmful}} + (1-\alpha) \cdot L_{\text{benign}}$ loss mixing weight (paper default) |
| `ablation_prob` | `float` | `0.5` | Per-step probability of activating the refusal-direction ablation hook |
| `lora_r` | `int` | `16` | LoRA rank (`0` disables LoRA) |
| `lora_alpha` | `int` | `32` | LoRA scaling factor |
| `merge_after_training` | `bool` | `True` | Whether to merge LoRA into base weights after training |

Plus all standard `TrainingArguments` (`output_dir`, `max_steps`,
`per_device_train_batch_size`, `learning_rate`, ...).

### Full example

```python
import tempfile
import torch
from datasets import Dataset
from safetune.harden import DeepRefusalTrainer, DeepRefusalConfig

# A refusal direction (unit vector) of shape (hidden_size,). In practice extract
# it with safetune.steer.extract_refusal_direction; here a random placeholder
# keeps the example self-contained.
refusal_direction = torch.randn(model.config.hidden_size)

def make_ds(texts):
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=16)
    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": enc["input_ids"],
    })
    return ds.with_format("torch")  # yield tensors, not lists

harmful_ds = make_ds(["How do I build a bomb?", "Explain how to hack a bank."])
benign_ds  = make_ds(["Write a poem about spring.", "Summarize the water cycle."])

with tempfile.TemporaryDirectory() as out_dir:
    config = DeepRefusalConfig(
        output_dir=out_dir,
        max_steps=1,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        alpha=0.2, ablation_prob=0.5, lora_r=8, lora_alpha=16,
        merge_after_training=False,
        report_to="none", use_cpu=True,
    )
    trainer = DeepRefusalTrainer(
        model, config,
        refusal_direction=refusal_direction,
        harmful_dataset=harmful_ds,
        benign_dataset=benign_ds,
    )
    trainer.train()  # HF Trainer.train() — datasets came in via the constructor
```

### When to use

- **Best for:** LoRA FT with stochastic per-layer refusal-direction ablation hooks.
- **Trade-offs:** LoRA-based; needs a precomputed `refusal_direction` plus harmful and benign datasets (see the config fields above).

### Citation

```bibtex
@article{deeprefusal2025,
  title  = {Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction},
  author = {Xie, Yuanbo and others},
  author = {Xie, et al.},
  year   = {2025},
  note   = {EMNLP 2025, arXiv:2509.15202},
}
```

---

## AntibodyTrainer

`AntibodyTrainer` is a `transformers.Trainer` **subclass**, and
`AntibodyConfig` **extends `TrainingArguments`**. `model`, `args`
(an `AntibodyConfig`), and `train_dataset` are passed like any HF Trainer;
`harmful_dataset` and the `xi` coefficient are explicit constructor kwargs.
`sam_rho` and `sim_threshold` are `AntibodyConfig` fields (`xi` is **not** — it
is a trainer kwarg). Datasets go in the constructor — `train()` takes no
positional dataset.

### Signature

```python
AntibodyTrainer(
    model: PreTrainedModel,
    args: AntibodyConfig,          # extends TrainingArguments
    train_dataset: Dataset,
    *,
    harmful_dataset: Dataset | None = None,
    mode: str = "align",
    xi: float = 0.0,
    sam_rho: float | None = None,  # overrides args.sam_rho when set
    **trainer_kwargs,              # forwarded to transformers.Trainer
)
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to fine-tune |
| `args` | `AntibodyConfig` | required | `TrainingArguments` subclass carrying `sam_rho` / `sim_threshold` |
| `train_dataset` | `Dataset` | required | SFT data for the main loss |
| `harmful_dataset` | `Dataset \| None` | `None` (kwarg) | Harmful reference batch for the alignment defense |
| `xi` | `float` | `0.0` (kwarg) | Theorem 4.1 coefficient: $\lambda_t = \max\{0,\ \xi - \langle g_{\text{sharp}}, g_{\text{align}}\rangle / \|g_{\text{sharp}}\|^2\}$ |

### AntibodyConfig fields (on `args`)

| Field | Type | Default | Description |
|---|---|---|---|
| `sam_rho` | `float` | `0.05` | SAM flatness perturbation radius |
| `sim_threshold` | `float` | `0.0` | Cosine-similarity threshold for gradient attenuation |

Plus all standard `TrainingArguments` (`output_dir`, `max_steps`, ...).

### Full example

```python
import tempfile
from datasets import Dataset
from safetune.harden import AntibodyTrainer, AntibodyConfig

def make_ds(texts):
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=16)
    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": enc["input_ids"],
    })
    return ds.with_format("torch")

sft_ds     = make_ds(["Write a poem about spring.", "Summarize the water cycle."])
harmful_ds = make_ds(["How do I build a bomb?", "Explain how to hack a bank."])

with tempfile.TemporaryDirectory() as out_dir:
    config = AntibodyConfig(
        output_dir=out_dir,
        max_steps=1,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        sam_rho=0.05, sim_threshold=0.0,
        report_to="none", use_cpu=True,
    )
    trainer = AntibodyTrainer(
        model=model, args=config, train_dataset=sft_ds,
        harmful_dataset=harmful_ds, xi=0.0,
    )
    trainer.train()  # HF Trainer.train() — datasets came in via the constructor
```

Captures a harmful reference gradient and scales per-step gradients by
`1 - max(0, cos-sim)` — updates aligned with harmful fine-tuning are
attenuated.

### When to use

- **Best for:** SAM-based flatness optimization + likelihood-ratio data reweighting.
- **Trade-offs:** requires a harmful reference batch.

### Citation

```bibtex
@article{antibody2026,
  title  = {Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence},
  author = {Nguyen, Quoc Minh and others},
  author = {Nguyen, et al.},
  year   = {2026},
  note   = {arXiv:2603.00498},
}
```

---

## LookAheadTrainer

### Signature

```python
LookAheadTrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    prefix_mode: str = "virtual",
    prefix_length: int = 6,
    **kwargs,
)
```

### LookAheadConfig fields

| Field | Type | Default | Description |
|---|---|---|---|
| `prefix_mode` | `str` | `"virtual"` | `"virtual"` or `"real"` prefix mode |
| `prefix_length` | `int` | `6` | Number of preview tokens |
| `prefix_token_ids` | `list[int] \| None` | `None` | Token IDs for the safe-opener prefix; derived from `prefix_text` when unset |
| `prefix_text` | `str` | `"Let's solve this problem."` | Text tokenized into the preview prefix when `prefix_token_ids` is unset |

### Full example

The runner `LookAheadTrainer` exposes `prefix_mode` and `prefix_length` only; the
prefix content comes from the `LookAheadConfig` defaults (`prefix_text` /
`prefix_token_ids`). Use the `safetune.harden` trainer with a custom
`LookAheadConfig` to set the prefix tokens explicitly.

```python
from safetune.runner import harden

trainer = harden.LookAheadTrainer(model, tokenizer, prefix_mode="virtual", prefix_length=6)
trainer.train(ds)
```

Prepends a fixed safe-opener prefix to every training example so SFT cannot
shift the aligned model's refusal-trigger token distribution.

### When to use

- **Best for:** anchoring safety alignment during SFT via answer-prefix preview.
- **Trade-offs:** effectiveness hinges on one fixed safe-opener prefix generalizing across whatever harmful fine-tuning attacks you're anchoring against; customizing the prefix text/tokens (not just its length/mode) requires dropping to the low-level `safetune.harden` trainer with a custom `LookAheadConfig`.

### Citation

```bibtex
@article{lookahead2026,
  title  = {LookAhead Tuning: Safer Language Models via Partial Answer Previews},
  author = {Liu, Kangwei and others},
  author = {Liu, et al.},
  year   = {2026},
  note   = {WSDM 2026, arXiv:2503.19041},
}
```
