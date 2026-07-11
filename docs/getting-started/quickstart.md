# Quick Start

SafeTune is a **library of safety methods**, not a pipeline — you
pick *one* method per task. This page shows the same task (running a
train-time defence, or `harden`) through all three entry points: Python, CLI,
and YAML. For the full reference see [Usage](../user-guide/usage.md).

New here? The fastest thing to run is the no-training STEER demo:

```bash
python examples/quickstart/quickstart.py
```

It extracts a refusal direction from contrast prompts, ablates it live on
`Qwen/Qwen2.5-0.5B-Instruct`, and prints the change — the full STEER path, end
to end, in about 30 seconds on CPU.

---

## 1. Python API

The `safetune.runner` package wraps each pillar in a uniform Trainer interface.
A harden trainer **replaces** your `transformers.Trainer` — it *is* the
fine-tuning:

```python
from safetune.runner import harden

# SafeGrad projects the task gradient off the alignment gradient each step,
# so fine-tuning on (possibly contaminated) data does not erode safety.
trainer = harden.SafeGradTrainer(model, tokenizer)
trainer.train(train_dataset, safety_dataset=safety_data)
```

Every other pillar follows the same shape — for example, inference-time
steering with no training:

```python
from safetune.runner import steer

trainer = steer.RefusalDirectionTrainer(model, tokenizer)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)
```

## 2. CLI

After install, the `safetune` command is available:

```bash
safetune list           # print every method, grouped by pillar

# Train with SafeGrad (the default harden method)
safetune train --model Qwen/Qwen2.5-0.5B-Instruct --algo safegrad --epochs 3

# Score a model on a safety benchmark
safetune eval --model Qwen/Qwen2.5-0.5B-Instruct --dataset harmbench
```

Run `safetune --help` for the full flag list, or see
[Usage → CLI](../user-guide/usage.md#1-cli).

## 3. YAML config

Declare all flags in a YAML file and pass it with `--config`. Any key that is
not a standard field is forwarded as a keyword argument to the trainer, so
method-specific hyperparameters live in the same file:

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

---

## Next steps

- [Examples](../examples/index.md) — a runnable script and notebook per pillar.
- [Usage](../user-guide/usage.md) — full CLI, YAML, and Python API reference.
- [Getting started](index.md) — the decision guide for picking a method.
