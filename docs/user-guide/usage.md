# Usage: CLI · YAML · Python

SafeTune can be used in three ways: command-line, YAML configuration, or
Python API. Pick the one that fits your workflow.

---

## 1. CLI

After `pip install safetune`, the `safetune` command is available:

```bash
safetune --help
safetune list           # print every available method, grouped by pillar
```

### Commands

| Command | Pillar | What it does |
|---|---|---|
| `safetune train` | Harden | Run a train-time defence |
| `safetune eval` | Evaluate | Score a model on safety benchmarks |
| `safetune patch` | Recover | Apply weight-space recovery |
| `safetune unlearn` | Unlearn | Prepare an unlearning trainer |
| `safetune list` | — | List all available method aliases |

### Examples

```bash
# Train with SafeGrad (default method)
safetune train --model Qwen/Qwen2.5-0.5B-Instruct --algo safegrad --epochs 3

# Train with Lisa on a custom HF dataset
safetune train --model Qwen/Qwen2.5-0.5B-Instruct --algo lisa \
               --train-dataset openai/gsm8k --train-split train

# Evaluate a model
safetune eval --model Qwen/Qwen2.5-0.5B-Instruct --dataset harmbench

# Patch a drifted model with RESTA
safetune patch --model ./drifted-model --algo resta --base ./base-model

# Unlearn with RMU
safetune unlearn --model ./model --algo rmu

# Load all flags from a YAML config (explicit CLI flags override YAML)
safetune train --config run.yaml --epochs 5
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--model` | required | Model path or HF hub ID |
| `--algo` | `safegrad` | Method alias — run `safetune list` to see all |
| `--config` | — | Path to a YAML config file; values act as parser defaults |
| `--dataset` | — | Comma-separated benchmarks for `eval` |
| `--train-dataset` | `beavertails` | Training dataset: `beavertails` or any HF dataset id |
| `--train-split` | — | Split to load from `--train-dataset` (default resolved per-dataset: `30k_train` for beavertails, else `train`) |
| `--output` | `./results` | Output / checkpoint directory |
| `--epochs` | `1` | Training epochs |
| `--batch-size` | `1` | Per-device batch size |
| `--lr` | `5e-5` | Learning rate |
| `--base` | — | Base model path (for `patch`) |
| `--aligned` | — | Aligned model path (for `patch`) |
| `--alpha` | — | Strength for `patch` methods (the trainer's `alpha`); unset: the method's default |
| `--precision` | `None` | `fp16` / `bf16` / `fp32`; unset: the runtime dtype (bf16 where supported) |
| `--wandb` | off | Enable WandB logging |

---

## 2. YAML Configuration

All CLI flags can be declared in a YAML file and passed via `--config`. Any
key that doesn't match a standard field is forwarded as a keyword argument
directly to the trainer, so you can set method-specific hyperparameters in the
config file.

### Minimal example

```yaml
# run.yaml
algo: lisa
model: Qwen/Qwen2.5-0.5B-Instruct
epochs: 3
batch_size: 4
train_dataset: beavertails
train_split: 30k_train
output: ./results/lisa

# Method-specific kwargs — forwarded to LisaTrainer(...)
lisa_rho: 0.2
lisa_warmup_steps: 20
```

```bash
safetune train --config run.yaml
safetune train --config run.yaml --epochs 5   # explicit flags override YAML
```

### Loading from Python

```python
from safetune.config import SafeTuneConfig

cfg = SafeTuneConfig.from_yaml("run.yaml")
print(cfg.algo)          # "lisa"
print(cfg.method_kwargs) # {"lisa_rho": 0.2, "lisa_warmup_steps": 20}

# Get a flat dict suitable for **kwargs into any Trainer constructor
kw = cfg.as_trainer_kwargs()

# Convert to an argparse.Namespace (all fields + method_kwargs flattened)
ns = cfg.to_namespace()
```

### `SafeTuneConfig` reference

| Field | Default | Notes |
|---|---|---|
| `command` | `"train"` | `train` / `eval` / `patch` / `unlearn` |
| `algo` | `"safegrad"` | Method alias |
| `model` | `""` | HF hub id or local path |
| `base` | `None` | Base model path (patch) |
| `aligned` | `None` | Aligned model path (patch) |
| `output` | `"./results"` | Checkpoint output dir |
| `epochs` | `1` | Training epochs |
| `batch_size` | `1` | Per-device batch size |
| `lr` | `5e-5` | Learning rate |
| `precision` | `None` | `fp16` / `bf16` / `fp32`; `None`: the runtime dtype (bf16 where supported) |
| `optimizer` | `"adamw_torch"` | Optimizer name |
| `logging_steps` | `10` | Log every N steps |
| `train_dataset` | `"beavertails"` | `beavertails` or HF dataset id |
| `train_split` | `None` | Dataset split (`None` resolves per-dataset in the CLI: `30k_train` for beavertails, else `train`) |
| `dataset` | `None` | Benchmarks for eval (comma-separated) |
| `drift_task` | `None` | Utility drift task (`gsm8k`, `code`, …) |
| `method_kwargs` | `{}` | Any unknown YAML key lands here |

---

## 3. Python API

### Quick reference

```python
from safetune.runner import harden, recover, unlearn, steer
from safetune.interpret import safety_circuit_info
from safetune.evaluate import evaluate

# Harden — train-time defence (replaces your SFT trainer)
trainer = harden.SafeGradTrainer(model, tokenizer)
trainer.train(train_dataset, safety_dataset=safety_data)

# Recover — weight-space patching (no training, ~30 seconds)
trainer = recover.ReStaTrainer(drifted, base_model=base, aligned_model=aligned)
patched = trainer.apply()

# Unlearn — forget-set training
trainer = unlearn.RMUTrainer(model)
clean = trainer.unlearn(forget=forget_batches, retain=retain_batches)

# Steer — inference-time steering
trainer = steer.RefusalDirectionTrainer(model, tokenizer)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

# Interpret — locate safety circuits
circuit = safety_circuit_info(model, tokenizer, harmful, harmless)
print(circuit.layer_suggestions.layer_subset)

# Evaluate — measure safety
results = evaluate(model, tokenizer=tokenizer, benchmarks=["harmbench"], judge="wildguard")
print(f"Refusal rate: {results['harmbench']['refusal_rate']:.1%}")
```

### Runner API

The `safetune.runner` package wraps each pillar in a uniform Trainer interface:

```python
from safetune.runner import harden, recover

# Harden: train → eval → save
trainer = harden.LisaTrainer(model, tokenizer, lisa_rho=0.1)
out_path = trainer.train(dataset, out_dir="./lisa")
metrics = trainer.eval("lisa_run", out_path)
trainer.save_results(metrics, variant="default")

# Recover: apply → save checkpoint → eval → save
trainer = recover.CThetaTrainer(
    drifted, base_model=base, aligned_model=aligned, strength=1.2
)
patched = trainer.apply()                       # returns the patched model
path    = trainer.save_checkpoint(patched, tokenizer, "ctheta_v1")
metrics = trainer.eval("ctheta_v1", path)
trainer.save_results(metrics, variant="strength=1.2")
```

`apply()` contract for recover trainers:
- Returns the patched model (may mutate `self.model` in-place or return a new object).
- Never writes to disk — the caller calls `save_checkpoint()` separately.
- Accepts method-specific overrides as keyword arguments.

See the full runner examples in `examples/runner/`.

---

## 4. Extending the method registry

The CLI resolves algo names to trainer classes via
`safetune/runner/_registry.py`. You can register additional methods at runtime
without editing any library file:

```python
from safetune.runner._registry import register_harden, register_recover, register_unlearn

# MyTrainer must be importable from safetune.runner.harden
register_harden("mymethod", "MyTrainer")

# Or for recover / unlearn:
register_recover("myrecover", "MyRecoverTrainer")
register_unlearn("myunlearn", "MyUnlearnTrainer")
```

After registration, `safetune train --algo mymethod` and
`safetune list` both reflect the new entry.

See [dev-runbook.md](../community/dev-runbook.md) for how to wire up a new
method end-to-end (implementation file → `__init__.py` re-export → registry entry).

---

## 5. Notebooks

Each pillar has a Colab-ready notebook under `examples/notebooks/`:

```bash
# Run locally
python examples/quickstart/quickstart.py

# Or open the notebook in Colab
# examples/notebooks/steer_demo.ipynb
```

See `examples/README.md` for the full matrix of scripts and notebooks.
