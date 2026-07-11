# Installation

SafeTune will be published on PyPI with the 1.0.0 release; you can also install from source (below). The core library installs cleanly on CPU; the
GPU-heavy backends (vLLM, Unsloth, TransformerLens) are optional extras you add
only when a method needs them.

## Requirements

- Python ≥ 3.12
- PyTorch (installed automatically as a core dependency)

## Install from PyPI

```bash
pip install safetune
```

This pulls the full core stack — Transformers, PEFT, TRL, Datasets, the
evaluation metrics, and the CLI. Every method's *core* implementation is
covered; only the faster GPU backends are held back as extras (see below).

## Install from source

```bash
git clone https://github.com/Lexsi-Labs/SafeTune.git
cd SafeTune && pip install -e .
```

Editable installs are the right choice if you plan to add a method or extend
the runner registry — see the [dev runbook](../community/dev-runbook.md).

## Verify the install

```python
import safetune
print(safetune.__version__)
```

The core imports cleanly on a CPU-only machine, so this works without a GPU.

## Optional GPU extras

Extras are declared under `[project.optional-dependencies]` in `pyproject.toml`
and installed with the `pip install "safetune[extra]"` syntax:

```bash
# Faster steering / eval backend
pip install "safetune[vllm]"

# vLLM with the activation-lens hook backend
pip install "safetune[vllm-lens]"

# Interpret pillar: TransformerLens-based circuit analysis
pip install "safetune[interpret]"

# Unsloth-accelerated fine-tuning for the harden pillar
pip install "safetune[unsloth]"

# Extra text-similarity metrics (BERTScore, sentence-transformers, CodeBLEU)
pip install "safetune[text-metrics]"

# Plotting helpers for the notebooks
pip install "safetune[viz]"
```

| Extra | Adds | Use it for |
|---|---|---|
| `vllm` | vLLM | faster steering and evaluation |
| `vllm-lens` | vLLM + vLLM-Lens | vLLM hook-based activation steering |
| `interpret` | TransformerLens | EAP / circuit discovery in `interpret` |
| `unsloth` | Unsloth | accelerated train-time hardening |
| `text-metrics` | BERTScore, sentence-transformers, CodeBLEU | richer utility metrics |
| `viz` | matplotlib, seaborn | notebook plots |
| `dev` | linters, pytest stack | contributing to SafeTune |
| `docs` | MkDocs Material stack | building these docs |

Combine extras in one call, e.g. `pip install "safetune[vllm,interpret]"`.

## Next steps

- [Quick Start](quickstart.md) — a runnable example per entry point.
- [Getting started](index.md) — pick the right method for your task.
- [Examples](../examples/index.md) — scripts and notebooks per pillar.
