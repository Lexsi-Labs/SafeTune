# Hyperparameter Reference

A single, scannable page for every tunable setting in SafeTune: the process-wide
`safetune.configure()` runtime settings, the parameters every trainer in a pillar
shares, and a quick-reference table for all 75 methods across Harden, Recover,
Unlearn, and Steer. Each method's row links to its full write-up — signature,
worked example, "when to use," and citation — in the [User Guide](../user-guide/index.md);
this page exists so you can compare parameters *across* methods without opening
75 separate pages, and so the runtime settings (which don't live on any one
method's page) get documented field-by-field in one place.

## How to read the tables

- **Default** is the value if you never pass that keyword argument.
- Every trainer's constructor also accepts `model`, and Harden/Steer additionally
  `tokenizer` — these are omitted from the per-method tables below since they're
  identical everywhere; see [Shared parameters](#shared-parameters-per-pillar).

---

## Runtime configuration — `safetune.configure()`

`safetune.configure(**settings)` sets **process-wide** settings (`RuntimeConfig`);
`safetune.get_config()` reads them back. Resolution order everywhere a setting is
consulted: an explicit function/trainer argument wins over `configure()` (or a
YAML config's `runtime:` block), which wins over a `SAFETUNE_*` environment
variable, which wins over the hard default below. An unrecognized setting name
raises `TypeError` (with a "did you mean" suggestion).

```python
import safetune
safetune.configure(max_prompts=20, eval_limit=20, eval_strict=False)
```

### Hardware

| Setting | Default | Significance |
|---|---|---|
| `device` | `"auto"` | `"auto"` picks CUDA, then Apple MPS, then CPU. Force a specific device (`"cpu"`, `"cuda:1"`) to override auto-detection. |
| `dtype` | `"auto"` | `"auto"` is bf16 where the hardware supports it natively (CUDA with bf16 support, Apple Silicon on macOS 14+), else fp16 on CUDA or fp32 elsewhere. |
| `eval_backend` | `None` (env `SAFETUNE_EVAL_BACKEND`) | `None` uses vLLM when it's installed, else the plain `transformers` ("hf") backend. Force one explicitly with `"vllm"` or `"hf"`. |
| `seed` | `None` (unseeded) | Sets the global random seed for reproducible runs. |

### Generation lengths (per purpose)

| Setting | Default | Significance |
|---|---|---|
| `safety_max_new_tokens` | `512` | Max new tokens generated per response in `trainer.evaluate()`'s safety benchmarks, the standalone `evaluate()`, and `generate_responses()`. The main lever on evaluation wall-clock time. |
| `advbench_max_new_tokens` | `128` | Same, for the AdvBench generation inside `trainer.evaluate()`. |
| `steer_max_new_tokens` | `256` | Generation length for Steer's `eval_live()` / `eval_vllm()`. |
| `forget_max_new_tokens` | `256` | Generation length when building a HarmBench-sourced forget set from a drifted model's own completions. |

### Batch sizes and sample counts

| Setting | Default | Significance |
|---|---|---|
| `gen_batch_size` | `8` | Batch size for the plain HF generation paths: `evaluate()`, `generate_responses()`, forget-set generation. |
| `steer_batch_size` | `4` | Batch size for Steer's `eval_live()`. |
| `lm_eval_batch_size` | `None` | Resolves to `"4"` on the "hf" backend or `"auto"` on vLLM. |
| `max_prompts` | `None` (every prompt) | Per-benchmark prompt cap for the safety suite — the main lever for a faster evaluation pass during development. |
| `eval_limit` | `None` (env `SAFETUNE_EVAL_LIMIT`, every example) | The lm-eval `--limit` flag — caps how many examples each utility benchmark (gsm8k, humaneval, mbpp, wikitext, ifeval, ...) scores, independent of `max_prompts` (which only caps the safety benches). |
| `harden_data_n` | `256` | Size of the contamination/refusal sets several Harden methods build automatically when no explicit `safety_dataset`/`harmful_dataset` is passed. |
| `calib_n` | `256` | Prompts per side (harmful/harmless) for Recover and Steer's default calibration sets. |
| `max_len` | `None` | Training tokenization length used by the CLI and several methods' auto-built datasets. `None` sizes it from the chat template for QA-shaped data, or 256 for fixed-length paths. |

### vLLM / lm-eval

| Setting | Default | Significance |
|---|---|---|
| `gpu_memory_utilization` | `None` (env `SAFETUNE_GPU_MEM`) | Fraction of GPU memory vLLM reserves up front. `None` lets each call site use its own default; set this explicitly to share a GPU with other workloads. |
| `max_model_len` | `4096` | vLLM's max context length. |
| `tensor_parallel_size` | `1` | vLLM tensor-parallel degree — raise this on a multi-GPU host to split a large model's weights across GPUs. |
| `lm_eval_timeout` | `7200` (2 hours) | Wall-clock timeout per lm-eval task group (there are up to three — chat, code, wiki — so a full `eval_utility()` call can run for several times this). Sized for a complete, unhurried audit; lower it for faster-failing evaluation during development. |

### Judges and scoring

| Setting | Default | Significance |
|---|---|---|
| `judges` | `{}` | Per-judge overrides (e.g. `{"orbench": {"max_model_len": 4096}}`), merged over the built-in judge registry. |
| `advbench_refusal_prefixes` | the built-in refusal-phrase tuple | The phrases `advbench_scorer="prefix"` looks for at the start of a response. Replace the tuple to count other refusal wording. |
| `advbench_scorer` | `"prefix"` | `"prefix"`: the response (minus any `<think>` block) starts with a standard refusal phrase. `"gcg"`: a looser match — any of a 29-phrase GCG test set appears anywhere in the response. |
| `orbench_in_safety_mean` | `False` | `False`: OR-Bench's over-refusal and toxic splits are reported separately and `safety_mean()` only averages genuine harm benchmarks. `True`: a single combined OR-Bench score is averaged into `safety_mean` with the rest. |
| `eval_strict` | `True` | `True`: `evaluate()` and every trainer's `.evaluate()` raise when a benchmark fails to load, generate, or score. `False`: that benchmark's metrics are recorded as an error entry and the run continues with the rest — useful when a gated benchmark shouldn't block an otherwise-successful run. |

### LoRA adapter defaults

Used by every Harden trainer that wraps the model in a LoRA adapter, and by
Unlearn's NPO/FLAT/SimDPO.

| Setting | Default | Significance |
|---|---|---|
| `lora_r` | `16` | LoRA rank. |
| `lora_alpha` | `32` | LoRA scaling factor. |
| `lora_dropout` | `0.05` | Dropout inside the LoRA adapter. |
| `lora_target_modules` | `("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")` | Which linear layers receive a LoRA adapter — the standard attention + MLP projection set. |

### Reproducing earlier results

A small set of `legacy_*` settings restore specific behavior from before a
correctness fix, for reproducing results generated on an earlier version. All
default to `False`.

| Setting | `True` restores |
|---|---|
| `legacy_beavertails_splits` | TAR's adversary set as `unsafe[n:2n]`, which overlaps the contamination set |
| `legacy_steer_layers` | Steer layer defaults as fixed absolute indices, instead of scaling proportionally to model depth |
| `legacy_cast_gate` | CAST's gate fitted on raw prompts, with the first prompt of a batch deciding for the whole batch |
| `legacy_spectral_monitor` | `SpectralEntropyMonitor` computed on raw prompts, including the attention-sink token in the SVD |
| `legacy_ga_forget_clip` | `GradientAscentTrainer` / `GradDiffTrainer`'s earlier default `forget_clip=0.5` |
| `legacy_constrained_sft` | `ConstrainedSFTTrainer` trains without its reference model by default (plain SFT) |
| `legacy_alphasteer_layers` | AlphaSteer's earlier, offset layer-to-decoder mapping |
| `legacy_derta` | DeRTa's MLE loss also runs over the harmful prefix, as a second loss term |
| `legacy_resta_drop_rate` | ReSta's DARE drop rate `0.9` instead of the current `0.3` |

### `datasets`

`configure(datasets={"harmbench": "./my_file.jsonl"})` points a built-in dataset
short name at another source — an HF id, a local `.jsonl`/`.json`/`.csv`/`.parquet`
file or directory, or a URL. Every loader that reads that name then reads the
override. Pass `datasets=None` to clear back to the built-in sources. See
[Benchmarks](../user-guide/evaluate/benchmarks/index.md) for the full dataset list.

---

## Shared parameters (per pillar)

Every method within a pillar inherits these from its pillar's base class —
listed once here rather than repeated 75 times below.

### Harden

```python
def __init__(self, model=None, tokenizer=None, *,
             model_id: str = None, epochs: int = 1, batch_size: int = 4, lr: float = 1e-4,
             bf16: bool | None = None, fp16: bool | None = None, wandb: bool = False,
             optimizer: str = "adamw_torch", logging_steps: int = 10,
             results_dir: str = None, drift_task: str = None, **kwargs)
```

| Param | Default | Significance |
|---|---|---|
| `epochs` | `1` | Training epochs. |
| `batch_size` | `4` | Per-device training batch size. |
| `lr` | `1e-4` | Optimizer learning rate. |
| `bf16` / `fp16` | `None` / `None` | `None` on both derives precision from the runtime (`dtype="auto"`). |
| `wandb` | `False` | Enables Weights & Biases reporting on the underlying training run. |
| `optimizer` | `"adamw_torch"` | Any optimizer name HF `TrainingArguments.optim` accepts. |
| `logging_steps` | `10` | Training-progress logging frequency. |
| `drift_task` | `None` | The capability-anchor domain (e.g. `"gsm8k"`) used when `.evaluate()` is called without an explicit `domain=`. |

**Verb**: `.train(train_dataset, out_dir=None, **kwargs) -> str` (returns the
checkpoint path). Every method also accepts `dataset_config=`, `dataset_split=`,
`safety_dataset_config=`, `safety_dataset_split=`, letting `train_dataset`/
`safety_dataset` be a source string (HF id, local file, dataset folder) instead
of a pre-tokenized `Dataset` — see
[Data Selection](../user-guide/harden/data-selection.md).

### Recover

```python
def __init__(self, model=None, *, model_id=None, results_dir: str = None,
             drift_task: str = None, **kwargs)
```

No tokenizer, no training hyperparameters — Recover is weight-space and
training-free for every method except `SCRUBTrainer`. **Verb**:
`.apply(**kwargs) -> model` (mutates `self.model` in place and/or returns a new
model object; call `.save_checkpoint(...)` separately to write it to disk).
`.evaluate(model_or_path, domain=)` accepts either a path or a live model object.

### Unlearn

```python
def __init__(self, model=None, *, model_id=None, results_dir: str = None,
             drift_task: str = None, upcast: bool = True, **kwargs)
```

| Param | Default | Significance |
|---|---|---|
| `upcast` | `True` | Trains fp16/bf16 model weights in fp32 for the duration of unlearning. The unlearn loops step a plain optimizer directly on the model's weights, and low-precision training there can silently produce little or no change in the model. Set `False` only to save memory once you've confirmed results still hold for your case. |

**Verb**: `.unlearn(forget, retain, **kwargs) -> model` — returns a model object
with no automatic checkpoint save, so call `.save_checkpoint(model, tokenizer, name) -> path`
yourself afterward. `forget`/`retain` accept a tokenized HF `Dataset`, a
`DataLoader`, or a plain list of `{input_ids, attention_mask, labels}` batches.
Every method automatically checks the model's weights after unlearning and
raises if any became invalid, so an unstable run surfaces immediately rather
than producing a silently broken checkpoint.

### Steer

```python
def __init__(self, model=None, tokenizer=None, *, model_id=None,
             results_dir: str = None, drift_task: str = None, **kwargs)
```

**Verb**: `.calibrate(harmful=None, harmless=None, *, calib_n=None, out_dir=None, **kwargs)`,
which returns `(wrapped_model, out_dir)`. The 4 decoding-steering methods
additionally expose a public `make_processor(...)`, whose result passes
directly to HF `generate(logits_processor=[...])`.

---

## Harden — 26 methods

Full guide: [Harden overview](../user-guide/harden.md).

### Gradient surgery

| Method | Params (non-default) | Significance | Guide |
|---|---|---|---|
| `SafeGradTrainer` | `rho=1.0`, `kl_temperature=1.0`, `reference_model_path=None` | `rho`: gradient-surgery alignment weight. `reference_model_path`: frozen reference for the KL signal — falls back to the tokenizer's model id if omitted. | [Guide](../user-guide/harden/gradient-surgery.md) |
| `PlainSFTTrainer` | — | Undefended SFT baseline, no safety mechanism — the comparison baseline for benchmarking every other Harden method. | [Guide](../user-guide/harden/gradient-surgery.md) |

### Data shaping / alternation

| Method | Params (non-default) | Significance | Guide |
|---|---|---|---|
| `LisaTrainer` | `lisa_rho=0.1`, `lisa_warmup_steps=10`, `lisa_alignment_step=20`, `lisa_finetune_step=20` | Alternates each training step between a task batch and a safety batch; `lisa_rho` is the proximal constraint pulling each phase back toward the other. | [Guide](../user-guide/harden/data-shaping.md) |
| `SPPFTTrainer` | `sppft_mode="freeze"` (passed as a keyword argument) | `sppft_mode`: `"freeze"` (default) freezes the identified safety-layer parameters during fine-tuning; `"scale"` reduces their learning rate instead of freezing them outright. | [Guide](../user-guide/harden/data-shaping.md) |
| `LookAheadTrainer` | `prefix_mode="virtual"`, `prefix_length=6` | `prefix_mode`: `"virtual"` or `"token"` prefix representation for the lookahead signal. | [Guide](../user-guide/harden/data-shaping.md) |
| `STARDSSTrainer` | `use_kl_penalty=True`, `kl_scale=1.0` | Reweights training examples by how much they threaten safety, spending more gradient signal on the riskiest ones. | [Guide](../user-guide/harden/data-shaping.md) |
| `DeRTaTrainer` | `contamination_pairs=None`, `refusal_pairs=None` (train-time) | Trains refusal as a decoupled objective from task performance; builds its own pair data automatically if not supplied. | [Guide](../user-guide/harden/data-shaping.md) |
| `CSTTrainer` | `beta=0.1`, `max_len=None`, `include_uncensored_pairs=True` | Trains one checkpoint to serve two system-prompt behaviors via preference optimization; builds its training pairs from the harden contamination set automatically. | [Guide](../user-guide/harden/data-shaping.md) |

### Regularization

| Method | Params (non-default) | Significance | Guide |
|---|---|---|---|
| `AsFTTrainer` | `reg_lambda=1.0`, `aligned_model_path=None` | Anchors the fine-tune close to the original aligned weights. Set `aligned_model_path` for the regularization to take effect — it is the reference the method keeps the fine-tune close to. | [Guide](../user-guide/harden/regularization.md) |
| `SAPTrainer` | `grad_rate=0.1`, `v_update_step=0.05`, `contrastive_temperature=1.0` | Contrastive alignment with perturbation; automatically detects a contrastive-format safety dataset and adapts its loss accordingly. | [Guide](../user-guide/harden/regularization.md) |
| `SurgeryTrainer` | `sink_lambda=0.01` | Detects and mutes the attention-sink pattern that harmful fine-tuning data tends to exploit. | [Guide](../user-guide/harden/regularization.md) |
| `BoosterTrainer` | `perturb_scale=0.01`, `n_harmful_batches=8` | Simulates the harmful-gradient direction during training and perturbs weights in advance to blunt its effect. | [Guide](../user-guide/harden/regularization.md) |

### Representation perturbation

| Method | Params (non-default) | Significance | Guide |
|---|---|---|---|
| `VaccineTrainer` | `rho=2.0` | Adds worst-case perturbations to hidden representations during training, so the fine-tune can't find a low-loss shortcut that also erodes safety. | [Guide](../user-guide/harden/representation.md) |
| `TVaccineTrainer` | `rho=2.0`, `top_k_ratio=0.5` | A token-aware variant of Vaccine — perturbs the top `top_k_ratio` fraction of layers rather than uniformly. | [Guide](../user-guide/harden/representation.md) |

### Tamper-resistant

| Method | Params (non-default) | Significance | Guide |
|---|---|---|---|
| `RepNoiseTrainer` | `noise_alpha=1.0`, `noise_beta=0.1`, `noise_gamma=1.0` | Representation-noising defense against later re-fine-tuning attacks. Pass genuine unsafe data as `harmful_dataset=` at train time. | [Guide](../user-guide/harden/tamper-resistant.md) |
| `CTRAPTrainer` | (`safety_dataset=`/`harmful_dataset=` at train time) | Primes an embedding-collapse trigger that fires if the model is later fine-tuned on harmful data. | [Guide](../user-guide/harden/tamper-resistant.md) |
| `SEAMTrainer` | (`safety_dataset=`/`harmful_dataset=` at train time) | Maintains semantic alignment during fine-tuning across a benign/adversarial/alignment three-way data split. | [Guide](../user-guide/harden/tamper-resistant.md) |
| `DOORTrainer` | `beta=0.5`, `refusal_w=1.0`, `unlearn_w=1.0`, `base_model_path=None` | Joint refusal-reward and preference training. Trains from contamination and refusal data directly rather than a task dataset — `train_dataset` is not used. | [Guide](../user-guide/harden/tamper-resistant.md) |
| `TARTrainer` | `inner_steps=25`, `inner_lr=1e-4`, `lambda_tar=1.0` | Task-aware regularization with an inner-loop adversarial update at each outer step. | [Guide](../user-guide/harden/tamper-resistant.md) |
| `MARTTrainer` | `adv_model=None`, `seed_prompts=None`, `num_rounds=4`, `num_candidates=8`, `safety_threshold=0.5` | Co-evolves an adversarial model against the target during training. `adv_model` defaults to a copy of the target if not supplied; trains from `seed_prompts` and a reward signal rather than a task dataset. | [Guide](../user-guide/harden/tamper-resistant.md) |
| `DeepRefusalTrainer` | `layer_idx=-1`, `ablation_prob=0.5`, `alpha=0.2`, `lora_r=16` | Trains a refusal direction under ablation and distills it into LoRA weights, computing the direction internally from a harmful/harmless prompt contrast. | [Guide](../user-guide/harden/tamper-resistant.md) |
| `AntibodyTrainer` | `mode="both"`, `sam_rho=0.05`, `lambda_refusal=1.0`, `lambda_cap=10.0` | Two-stage defense: flatness alignment on harmful batches, then reweighting against a simulated adversary. `mode="both"` (default) runs both stages. | [Guide](../user-guide/harden/tamper-resistant.md) |

### Data selection

| Method | Params (non-default) | Significance | Guide |
|---|---|---|---|
| `SEALTrainer` | — | Selects which fine-tuning examples to actually train on, filtering out those most likely to erode safety. | [Guide](../user-guide/harden/data-selection.md) |
| `ConstrainedSFTTrainer` | `reference_model_path=None`, `use_reference=None` | Constrains which weight subspace the fine-tune can move in via a KL constraint against a reference model, used by default. | [Guide](../user-guide/harden/constrained-sft.md) |

### Pre-FT / weight-space

| Method | Params (non-default) | Significance | Guide |
|---|---|---|---|
| `LoXHardenTrainer` | `rank=8`, `extrapolation_factor=0.3`, `aligned_model_path=None` | Pre-extrapolates the base model's weights away from the region fine-tuning tends to erode, before training starts. Set `aligned_model_path` for the extrapolation to take effect. | [Guide](../user-guide/harden/pre-ft.md) |
| `SaLoRATrainer` | `rank=16`, `safety_rank=16`, `strength=1.0` | Projects LoRA updates so they never overwrite the subspace that encodes refusal — a lightweight addition alongside your existing LoRA fine-tune. | [Guide](../user-guide/harden/pre-ft.md) |

---

## Recover — 24 methods

Full guide: [Recover overview](../user-guide/recover.md). Most methods need
`base_model=`/`aligned_model=` — already-loaded model objects, not paths (the
exception is `AAQTrainer`, which takes path/HF-id strings). A number of methods
work from calibration prompts instead and need no reference model (noted per row).

### Whole-model

| Method | Reference model(s) | Key params | Significance | Guide |
|---|---|---|---|---|
| `TaskArithmeticTrainer` | `base_model`, `aligned_model` | `alpha=1.0` | Adds the safety vector `aligned − base` back to the drifted model, scaled by `alpha`. The simplest baseline — one hyperparameter to tune. | [Guide](../user-guide/recover/whole-model/task-arithmetic.md) |
| `PrePostMergeTrainer` | `pre_model` (object or path) | `alpha=0.5` | Interpolates toward the pre-fine-tune model. | [Guide](../user-guide/recover/whole-model/pre-post-merge.md) |
| `WiseFTTrainer` | `aligned_model` | `alpha=0.5` | Weight-space ensembling toward the aligned model. | [Guide](../user-guide/recover/whole-model/wise-ft.md) |
| `SOMFTrainer` | `aligned_model`, `base_model` | `mask_threshold=0.9` | Merges using a magnitude-based mask over the sign-overlapping weight regions between the drifted and aligned models. | [Guide](../user-guide/recover/whole-model/somf-merge.md) |

### Layer

| Method | Reference model(s) | Key params | Significance | Guide |
|---|---|---|---|---|
| `ReStaTrainer` | `base_model`, `aligned_model` | `alpha=1.0`, `dare=True`, `dare_drop_rate=None` (→0.3) | DARE-masked safety-vector restoration. Start with a lower `alpha` (0.1-0.25) on smaller models and sweep upward by evaluation score. | [Guide](../user-guide/recover/layer/resta.md) |
| `SafeLoRATrainer` | `aligned_state_dict`, `base_state_dict` (state dicts) | `alpha=0.5`, `threshold=0.5` | Safety-subspace decomposition and merge via LoRA. | [Guide](../user-guide/recover/layer/safe-lora.md) |
| `SafeMergeTrainer` | `base_model`, `aligned_model` | `threshold=0.35`, `alpha=0.5` | Fisher-weighted safety merge, gating which parameters are merged by an importance criterion. | [Guide](../user-guide/recover/layer/safe-merge.md) |
| `SafeDeltaTrainer` | `aligned_model` | `strength=0.1` | Targeted output-weight safety edit derived from the aligned model. | [Guide](../user-guide/recover/layer/safe-delta.md) |
| `QReSafeTrainer` | none (calibration-driven) | `mode="selective"`, `quant_bits=4`, `tau=0.6` | Selective quantization-aware safety recovery; `tau` is the safety-saliency threshold used to select which weights to quantize. | [Guide](../user-guide/recover/layer/qresafe.md) |
| `AAQTrainer` | `aligned_model_path`, `base_model_path` (strings) | `calibration_steps=10`, `lr=5e-6` | Adversarial alignment quantization with calibration. | [Guide](../user-guide/recover/layer/aaq.md) |
| `RepNoiseRecoverTrainer` | none (calibration-driven) | `noise_scale=0.01`, `subspace_rank=8` | Adds representation noise to the harm subspace identified from calibration inputs. | [Guide](../user-guide/recover/layer/repnoise-recover.md) |

### Low-rank

| Method | Reference model(s) | Key params | Significance | Guide |
|---|---|---|---|---|
| `LoXTrainer` | `base_model`, `aligned_model` | `rank=8`, `extrapolation_factor=0.3` | Adds a rank-limited, scaled safety delta between the aligned and base models. | [Guide](../user-guide/recover/low-rank/lox.md) |
| `LSSFTrainer` | `base_model`, `aligned_model` | `alpha=1.0`, `rank=8`, `eta=0.85` | Low-rank safety-subspace fusion; `eta` controls how much of the safety delta's singular-value spectrum is retained. | [Guide](../user-guide/recover/low-rank/lssf.md) |
| `SafetyVectorRestoreTrainer` | `aligned_model` | `alpha=1.0`, `rank=8` | Truncated-SVD safety-subspace injection derived from the aligned model. | [Guide](../user-guide/recover/low-rank/safety-vector-restore.md) |

### Neuron

| Method | Reference model(s) | Key params | Significance | Guide |
|---|---|---|---|---|
| `NLSRTrainer` | `donor_state` (state dict) | `blend=0.5` | Neuron-level safety restoration via a donor-state transplant. | [Guide](../user-guide/recover/neuron/nlsr.md) |
| `AntidoteTrainer` | none (calibration-driven) | `prune_fraction=0.005` | Prunes weights most associated with harmful outputs, identified from calibration prompts. | [Guide](../user-guide/recover/neuron/antidote-v1.md) |
| `AntidoteV2Trainer` | none (calibration-driven) | `global_prune_fraction=0.005`, `utility_floor=0.1` | Antidote with a utility floor and a budget limiting overlap between harm-salient and utility-salient weights. | [Guide](../user-guide/recover/neuron/antidote-v2.md) |
| `MSCPTrainer` | `aligned_state`, `base_state` (state dicts) | `coefficient=0.05` | Projects the model's weights to satisfy a safety constraint derived from the aligned/base state dicts. | [Guide](../user-guide/recover/neuron/mscp.md) |

### Saliency

| Method | Reference model(s) | Key params | Significance | Guide |
|---|---|---|---|---|
| `GradSelectiveRecoverTrainer` | `aligned_model` + `harmful_inputs` | `top_fraction=0.1` | Identifies harmful-gradient-salient weights and rolls them back to the aligned model's values. | [Guide](../user-guide/recover/saliency/grad-selective.md) |
| `OneShotSafetyPatchTrainer` | none (single prompt pair) | `top_fraction=0.05`, `lr=1e-4`, `num_steps=5` | A short gradient patch restricted to the most salient weights, derived from a single harmful/safe example pair — the lightest-weight method in this pillar. | [Guide](../user-guide/recover/saliency/one-shot-patch.md) |

### Circuit

| Method | Reference model(s) | Key params | Significance | Guide |
|---|---|---|---|---|
| `CThetaTrainer` | `base_model`, `aligned_model` | `strength=1.0`, `circuit_info` | Applies the safety delta only at weight locations identified by a `CircuitInfo` object from the Interpret pillar's `safety_circuit_info()` — zero collateral change to weights outside the identified circuit. | [Guide](../user-guide/recover/circuit/c-delta-theta.md) |

### Other

| Method | Reference model(s) | Key params | Significance | Guide |
|---|---|---|---|---|
| `PKETrainer` | `clean_model`, `toxic_model` | `top_k_neurons=50`, `num_steps=10` | Edits the top-k safety-relevant neurons toward the clean model's behavior. | [Guide](../user-guide/recover/neuron/pke.md) |
| `SafeReActTrainer` | `reference_model` | `probe_inputs=None` | Reactivates dormant safety neurons identified via probe inputs against the reference model. | [Guide](../user-guide/recover/neuron/safereact.md) |
| `SCRUBTrainer` | none (training batches instead) | `sgda_epochs=3`, `lr=5e-5`, `alpha=1.0` | The one Recover method that trains rather than purely editing weights — pass `retain=`/`forget=` batches to `.apply()`. | [Guide](../user-guide/recover/neuron/scrub-recover.md) |

---

## Unlearn — 6 methods

Full guide: [Unlearn overview](../user-guide/unlearn.md). `forget`/`retain` data:
`unlearn.load_unlearn_data(model_id, n=256)` returns `(forget_ds, retain_ds)` as
tokenized HF `Dataset`s.

| Method | Params (non-default) | Significance | Guide |
|---|---|---|---|
| `RMUTrainer` | `layer_id=7`, `max_num_batches=80`, `lr=5e-5`, `alpha=1200.0`, `steering_coeff=20.0` | Steers a chosen layer's activations toward a fixed random vector on forget data while anchoring retain-data activations to a frozen reference — trains only a few early layers' MLP weights, far fewer parameters than full-model unlearning. Note: constructing the lower-level `RMUConfig` directly uses `alpha=100.0` by default; `RMUTrainer`'s own default is `1200.0`. | [Guide](../user-guide/unlearn/rmu.md) |
| `NPOTrainer` | `variant="npo_grad_diff"`, `beta=0.1`, `num_epochs=5`, `lr=1e-5` | Negative preference optimization. `variant="npo_grad_diff"` (default) adds a retain-CE term alongside the forget term; `"npo"` uses the forget term alone. | [Guide](../user-guide/unlearn/npo.md) |
| `GradientAscentTrainer` | `epochs=5`, `max_steps=200`, `lr=1e-5`, `forget_clip=None` | Gradient ascent on the forget set. `forget_clip=None` (default) applies no cap on the forget loss. | [Guide](../user-guide/unlearn/gradient-ascent.md) |
| `GradDiffTrainer` | `epochs=5`, `max_steps=200`, `lr=1e-5` | `GradientAscentTrainer` with an added retain-CE term, always training full model weights rather than a LoRA adapter. | [Guide](../user-guide/unlearn/gradient-ascent.md) |
| `FLATTrainer` | `variant="flat_retain"`, `divergence="kl"`, `epochs=5`, `retain_coeff=1.0` | f-divergence-based forgetting. `divergence="kl"`, `"jensen_shannon"`, `"total_variation"`, and `"jeffrey"` are the recommended, stable choices. | [Guide](../user-guide/unlearn/flat.md) |
| `SimDPOTrainer` | `variant="simdpo_retain"`, `beta=0.1`, `epochs=5`, `retain_coeff=1.0` | Preference-optimization-based forgetting — builds `{chosen, rejected}` pairs from the forget set automatically. | [Guide](../user-guide/unlearn/simdpo.md) |

---

## Steer — 19 methods

Full guide: [Steer overview](../user-guide/steer.md). Steer is inference-time
only — no weight changes, so there's no `.train()`/`.apply()`/`.unlearn()` here,
only `.calibrate()` (see [Shared parameters](#steer)).

### Activation steering

| Method | Params (non-default) | Significance |
|---|---|---|
| `RefusalDirectionTrainer` | `layers=None`, `alpha=20.0` | Extracts and applies a refusal direction at the given layers. |
| `CAATrainer` | `target_layers=None`, `pool_method="mean"`, `multiplier=20.0` | Contrastive Activation Addition. |
| `LinearProbeGuardTrainer` | `layer=None`, `threshold=0.5` | Fits a linear probe to detect harmful activations and gates generation by `threshold`. |
| `SCANSTrainer` | — | Fits directly from calibration prompts at calibration time. |
| `STATrainer` | — | Requires no calibration data. |
| `CircuitBreakerRRTrainer` | `threshold=0.5` | Representation-rerouting circuit breaker. |
| `CASTTrainer` | `behavior_layers=None`, `alpha=1.0`, `per_prompt_gate=None` | Conditional activation steering. `per_prompt_gate` (default: on) gates and steers each prompt of a batch independently rather than letting one prompt's gate decide for the whole batch. |
| `AdaSteerTrainer` | `alpha=15.0`, `layers=None` | Adapts steering strength per input rather than applying a fixed strength uniformly. |
| `SafeSwitchTrainer` | `gate_layer=None`, `threshold=0.5` | Trains a small safety classifier to gate generation at a chosen layer. |
| `CircuitBreakerTrainer` | — | Requires no calibration data. |
| `RepBendTrainer` | — | Requires no calibration data. |
| `TARSteerTrainer` | — | Requires no calibration data. |
| `RRFAEnsembleTrainer` | — | Ensembles multiple refusal-direction-based safeguards. |
| `SafeSteerTrainer` | `alpha=15.0` | Extracts and applies a refusal direction as a single steering category. |
| `AlphaSteerTrainer` | `alpha=20.0`, `layers=None` | Per-layer steering vectors fit from harmful/benign activation contrasts. |

### Decoding steering

These methods blend or contrast logits with a second model at decode time and
require that second model to be supplied explicitly.

| Method | Params (non-default) | Significance |
|---|---|---|
| `SafeDecodingTrainer` | `alpha=2.0`, `window=5`, `expert_model` (required) | Blends base-model and safety-expert-model logits at decode time. |
| `ContrastiveDecodingTrainer` | `alpha=1.0`, `weak_model` (required) | Contrasts against a weaker reference model's logits to amplify the stronger model's safety behavior. |
| `ProxyTuningTrainer` | `proxy_model` (required), `alpha=1.0` | Applies a proxy model's logit deltas to steer decoding. |
| `NudgingTrainer` | `nudge_strength=1.0`, `safe_tokens=None`, `unsafe_tokens=None` | Nudges token probabilities toward safe tokens and away from unsafe ones. |

---

## Working examples

### Fast evaluation during development

When you're integrating SafeTune into a build or deployment pipeline, you want
to confirm a method runs correctly — real generation, real scoring — without
waiting for a full evaluation pass on every test run. Scale the safety suite
down for development, then run it at full scale for your actual pre-release
audit.

```python
import safetune
from safetune.runner import harden

# Development: cap prompts and generation length for a fast pass.
safetune.configure(max_prompts=20, eval_limit=20, safety_max_new_tokens=64)

trainer = harden.SafeGradTrainer(model, tokenizer)
before = trainer.evaluate(base_model_id, domain="support")
out_dir = trainer.train(train_dataset, out_dir="./dev_run")
after = trainer.evaluate(out_dir, domain="support")
```

```python
# Pre-release: reset to the full, unbounded defaults for the real audit.
safetune.configure(max_prompts=None, eval_limit=None, safety_max_new_tokens=512)
```

### Recovering a model already in production

A customer-facing model has been fine-tuned on new support data and safety
evaluation shows it has drifted — it now complies with requests it previously
refused. Retraining isn't an option on the current release schedule, so the
team repairs the deployed checkpoint directly.

```python
from safetune.runner import recover
from safetune.runner.utils import load_model_cpu, load_tok

drifted = load_model_cpu("acme-corp/support-assistant-v3")
base = load_model_cpu("meta-llama/Llama-3.1-8B")
aligned = load_model_cpu("meta-llama/Llama-3.1-8B-Instruct")

trainer = recover.TaskArithmeticTrainer(drifted, base_model=base, aligned_model=aligned, alpha=1.0)
repaired = trainer.apply()
ckpt = trainer.save_checkpoint(repaired, load_tok("acme-corp/support-assistant-v3"), "support-assistant-v3-repaired")
metrics = trainer.eval("support-assistant-v3-repaired", ckpt)
```

Start with `TaskArithmeticTrainer` and sweep `alpha` against your evaluation
suite before moving to a more targeted method — it has the fewest moving parts
and the fastest turnaround for a first repair.

### Removing a specific capability without a broader safety pass

A model fine-tuned on internal support transcripts picked up a tendency to
give specific regulated financial guidance it isn't authorized to give, while
the rest of its behavior is unaffected. The goal is to remove that one
capability precisely, not to run a general safety repair.

```python
from safetune.runner import unlearn

model, tok = load_model("acme-corp/support-assistant-v3"), load_tok("acme-corp/support-assistant-v3")
forget_ds, retain_ds = unlearn.load_unlearn_data("acme-corp/support-assistant-v3", n=256)

trainer = unlearn.RMUTrainer(model, layer_id=7)
unlearned = trainer.unlearn(forget_ds, retain_ds)
ckpt = trainer.save_checkpoint(unlearned, tok, "support-assistant-v3-unlearned")
metrics = trainer.eval("support-assistant-v3-unlearned", ckpt)
```

`RMUTrainer` trains the fewest parameters of the methods in this pillar, which
makes it the first one to try and the easiest to evaluate and roll back.
`GradientAscentTrainer` and `NPOTrainer` go further and train more of the
model if `RMUTrainer` doesn't suppress the target capability enough — check
both the forget-set score and the retained utility metrics before escalating.
