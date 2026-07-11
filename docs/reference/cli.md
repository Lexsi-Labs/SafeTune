# CLI Reference

SafeTune ships a single console script, `safetune`, that dispatches to each safety pillar. The command is **positional** — `safetune <command> [flags]`, not a set of subcommand parsers — so global flags apply to every command. The five commands are `train` (Harden), `eval` (Evaluate), `patch` (Recover), `unlearn` (Unlearn), and `list`. The CLI covers the common train/recover/unlearn/evaluate paths; steer and interpret, and the programmatic-only harden methods, live in the [Python API](api/index.md).

## Installation

```bash
pip install safetune
safetune list
```

`safetune list` prints every registered method grouped by pillar (Harden, Recover, Unlearn) and needs no model — use it to discover the exact `--algo` alias for a method.

## Command Overview

| Command | Pillar | Description |
|---|---|---|
| `train` | Harden | Train-time defense: fine-tune a model with a harden trainer and save the hardened weights. |
| `patch` | Recover | Weight-space recovery: apply a recover method to a drifted model, optionally using base/aligned references. |
| `unlearn` | Unlearn | Forget-set training: build an unlearn trainer for a model, then call it from Python. |
| `eval` | Evaluate | Run safety/utility benchmarks against a model and print per-benchmark metrics. |
| `list` | — | Print every available method per pillar. No model required. |

## safetune train

Harden a model by fine-tuning it with a train-time defense. The trainer named by `--algo` is looked up in `HARDEN_REGISTRY`; the model is loaded, the training dataset (`--train-dataset` / `--train-split`, default BeaverTails `30k_train`) is tokenized, and the hardened weights are written to `--output`.

```bash
safetune train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --algo lisa \
  --train-dataset beavertails \
  --epochs 3 --batch-size 4 --lr 2e-5 \
  --output ./results/lisa
```

Relevant flags: `--algo`, `--model`, `--train-dataset`, `--train-split`, `--epochs`, `--batch-size`, `--lr`, `--precision`, `--output`, `--wandb`.

!!! note "Programmatic-only harden methods"
    A few harden methods — `cst`, `mart`, `deeprefusal`, `antibody` — need method-specific data/config that the uniform CLI train contract can't supply. Running e.g. `safetune train --algo cst` prints a hint pointing at the Python API (`from safetune.runner.harden import CSTTrainer`) and exits. Use these from Python; see the harden guide.

## safetune eval

Evaluate a model against one or more benchmarks and print each benchmark's metrics. Pass `--dataset` as a comma-separated list of benchmark names; omit it to run the default suite. `--eval-backend` selects the generation backend (`vllm` or `hf`); the default is auto — vLLM when installed, otherwise transformers.

```bash
safetune eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset harmbench,advbench \
  --eval-backend vllm
```

Relevant flags: `--model`, `--dataset`, `--eval-backend`.

## safetune patch

Recover safety in a drifted model via weight-space editing. The trainer named by `--algo` is looked up in `RECOVER_REGISTRY`. Many recover methods need reference checkpoints: `--base` supplies the base model and `--aligned` the aligned model used to extract the safety delta. The command applies the patch in memory and reminds you to call `trainer.save_checkpoint()` from Python to persist it.

```bash
safetune patch \
  --model ./drifted \
  --algo resta \
  --base ./base
```

Relevant flags: `--algo`, `--model`, `--base`, `--aligned`.

## safetune unlearn

Build an unlearn (forget-set training) trainer for a model. The trainer named by `--algo` is looked up in `UNLEARN_REGISTRY`. The CLI constructs the trainer and prints readiness; the actual forget/retain step runs in Python via `trainer.unlearn(forget=..., retain=...)`, since it needs the forget and retain sets.

```bash
safetune unlearn \
  --model ./model \
  --algo rmu \
  --epochs 1 --lr 5e-5
```

Relevant flags: `--algo`, `--model`, `--epochs`, `--batch-size`, `--lr`, `--precision`.

## safetune list

Print every registered method for each pillar (Harden, Recover, Unlearn) as `alias → TrainerClass`, followed by example invocations. Requires no `--model`.

```bash
safetune list
```

## Global Flags

All flags are global — they parse regardless of command, though not every command reads every flag. `--model` is required for every command except `list`.

| Flag | Default | Description |
|---|---|---|
| `--config` | — | Path to a YAML config file. Values are overridden by explicit CLI flags. |
| `--algo` | `safegrad` | Method (alias) within the pillar; see `safetune list`. |
| `--model` | — | Model path or HF hub ID. Required except for `list`. |
| `--dataset` | — | Dataset or comma-separated benchmarks for `eval`. |
| `--output`, `--output-dir` | `./results` | Output directory. |
| `--epochs` | `1` | Training epochs. |
| `--batch-size` | `1` | Batch size. |
| `--lr`, `--learning-rate` | `5e-5` | Learning rate / alpha. |
| `--base` | — | Base model path (for `patch`/recover). |
| `--aligned` | — | Aligned model path (for `patch`/recover). |
| `--precision` | `bf16` | Compute precision: `fp16`, `bf16`, or `fp32`. |
| `--train-dataset` | `beavertails` | Training dataset: `beavertails` or any HF dataset id. |
| `--train-split` | — | Split for `--train-dataset` (default: `30k_train` for beavertails, else `train`). |
| `--eval-backend` | auto | Generation backend for eval: `vllm` or `hf` (auto: vLLM if installed, else `hf`). |
| `--wandb` | off | Enable Weights & Biases logging. |

## YAML Configuration

Instead of passing every flag, describe a run declaratively with `--config`. The YAML maps onto `SafeTuneConfig`: recognized keys (`command`, `algo`, `model`, `base`, `aligned`, `output`, `epochs`, `batch_size`, `lr`, `precision`, `train_dataset`, `train_split`, `dataset`, `eval_backend`, ...) become standard fields, and any **unrecognized** key is forwarded to the trainer as a method-specific keyword argument (e.g. `lisa_rho`, `rank`, `inner_steps`).

```yaml
# lisa.yaml
command: train
algo: lisa
model: Qwen/Qwen2.5-0.5B-Instruct
epochs: 3
batch_size: 4
lr: 2e-5
train_dataset: beavertails
train_split: 30k_train
output: ./results/lisa
# method-specific kwargs — forwarded to the trainer
lisa_rho: 0.2
lisa_warmup_steps: 20
```

```bash
safetune --config lisa.yaml train
```

**Precedence:** `--config` values seed the parser defaults, and explicit CLI flags override them. So the config supplies the baseline and any flag on the command line wins:

```bash
# Uses lisa.yaml, but overrides the model on the command line
safetune --config lisa.yaml train --model other/model
```
