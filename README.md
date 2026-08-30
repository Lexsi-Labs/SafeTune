<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/safetune-compass-logo-dark.svg">
    <img src="docs/assets/safetune-compass-logo-light.svg" alt="SafeTune" width="360"/>
  </picture>
</p>

<h3 align="center">A library of LLM-safety methods. Pick the one that fits your task — and know exactly what it implements.</h3>

<p align="center">
  <a href="https://github.com/Lexsi-Labs/SafeTune/blob/main/CHANGELOG.md"><img src="https://img.shields.io/badge/version-0.1.0-5B3DD6.svg" alt="Version 0.1.0"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12%2B-blue.svg" alt="Python 3.12+"/></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-LSAL%20v1.2%20(source--available)-blue.svg" alt="License: LSAL v1.2"/></a>
</p>

<br>

SafeTune collects the many published methods for changing or measuring a
model's safety and puts them behind one consistent API. It is a **library, not a
pipeline**: each safety task has several methods that solve it by different
mechanisms, and you pick the one that fits — you don't chain them together.

## Install

```bash
pip install safetune            # from PyPI
# or, from source:
git clone https://github.com/Lexsi-Labs/SafeTune.git
cd SafeTune && pip install -e .
```

Requires Python ≥ 3.12 and PyTorch. The core library imports cleanly on CPU;
heavier extras (vLLM, Unsloth) install only when you ask for them.

## Run one in 60 seconds

```bash
python examples/quickstart/quickstart.py
```

This runs the inference-time **Steer** path end to end on a small open model:
it extracts a refusal direction from contrast prompts, ablates it live, and
prints how refusal behaviour changes — no training, no checkpoints.

## Examples and notebooks

Every intervention class also has a runnable script under
[`examples/`](examples/) — same code, terminal output instead of a browser;
see [Python Scripts](docs/examples/scripts.md) for the full list. The table
below is the notebook side: all 10 ship in
[`examples/notebooks/`](examples/notebooks/), each opens straight into a free
Colab runtime (no local install), and all default to
`Qwen/Qwen2.5-0.5B-Instruct`.

- **01–06 · Demos** — one per pillar, runs to completion with printed output.
  Start with `steer_demo`.
- **07–08 · Comparisons** — several methods run side by side on the same
  checkpoint, so you can see the trade-off directly.
- **09–10 · Advanced** — a live monitoring demo and the full six-pillar
  pipeline chained end to end.

| # | Notebook | Pillar | What it shows | GPU | Open |
|---|---|---|---|---|---|
| <sub>01</sub> | <sub>[`steer_demo`](examples/notebooks/steer_demo.ipynb)</sub> | <sub>Steer</sub> | <sub>extract a refusal direction and ablate it live — no training</sub> | <sub>No GPU</sub> | [<img src="https://colab.research.google.com/assets/colab-badge.svg" height="32" alt="Open In Colab">](https://colab.research.google.com/github/Lexsi-Labs/SafeTune/blob/main/examples/notebooks/steer_demo.ipynb) |
| <sub>02</sub> | <sub>[`recover_demo`](examples/notebooks/recover_demo.ipynb)</sub> | <sub>Recover</sub> | <sub>`ReStaTrainer` weight patching on a drifted model — no training</sub> | <sub>No GPU</sub> | [<img src="https://colab.research.google.com/assets/colab-badge.svg" height="32" alt="Open In Colab">](https://colab.research.google.com/github/Lexsi-Labs/SafeTune/blob/main/examples/notebooks/recover_demo.ipynb) |
| <sub>03</sub> | <sub>[`harden_demo`](examples/notebooks/harden_demo.ipynb)</sub> | <sub>Harden</sub> | <sub>`SafeGradTrainer` gradient-surgery fine-tune</sub> | <sub>GPU helps</sub> | [<img src="https://colab.research.google.com/assets/colab-badge.svg" height="32" alt="Open In Colab">](https://colab.research.google.com/github/Lexsi-Labs/SafeTune/blob/main/examples/notebooks/harden_demo.ipynb) |
| <sub>04</sub> | <sub>[`unlearn_demo`](examples/notebooks/unlearn_demo.ipynb)</sub> | <sub>Unlearn</sub> | <sub>`GradientAscentTrainer` removes a capability via forget/retain sets</sub> | <sub>GPU helps</sub> | [<img src="https://colab.research.google.com/assets/colab-badge.svg" height="32" alt="Open In Colab">](https://colab.research.google.com/github/Lexsi-Labs/SafeTune/blob/main/examples/notebooks/unlearn_demo.ipynb) |
| <sub>05</sub> | <sub>[`interpret_demo`](examples/notebooks/interpret_demo.ipynb)</sub> | <sub>Interpret</sub> | <sub>locate safety circuits and neurons from contrast prompts</sub> | <sub>No GPU</sub> | [<img src="https://colab.research.google.com/assets/colab-badge.svg" height="32" alt="Open In Colab">](https://colab.research.google.com/github/Lexsi-Labs/SafeTune/blob/main/examples/notebooks/interpret_demo.ipynb) |
| <sub>06</sub> | <sub>[`evaluate_demo`](examples/notebooks/evaluate_demo.ipynb)</sub> | <sub>Evaluate</sub> | <sub>benchmarks + red-team attacks + spectral entropy monitor</sub> | <sub>GPU helps</sub> | [<img src="https://colab.research.google.com/assets/colab-badge.svg" height="32" alt="Open In Colab">](https://colab.research.google.com/github/Lexsi-Labs/SafeTune/blob/main/examples/notebooks/evaluate_demo.ipynb) |
| <sub>07</sub> | <sub>[`steer_comparison`](examples/notebooks/steer_comparison.ipynb)</sub> | <sub>Steer</sub> | <sub>CAA vs RefusalDirection vs CAST vs AdaSteer, same checkpoint</sub> | <sub>GPU helps</sub> | [<img src="https://colab.research.google.com/assets/colab-badge.svg" height="32" alt="Open In Colab">](https://colab.research.google.com/github/Lexsi-Labs/SafeTune/blob/main/examples/notebooks/steer_comparison.ipynb) |
| <sub>08</sub> | <sub>[`recover_comparison`](examples/notebooks/recover_comparison.ipynb)</sub> | <sub>Recover</sub> | <sub>RESTA vs C-ΔΘ vs LoX, same drifted checkpoint</sub> | <sub>GPU helps</sub> | [<img src="https://colab.research.google.com/assets/colab-badge.svg" height="32" alt="Open In Colab">](https://colab.research.google.com/github/Lexsi-Labs/SafeTune/blob/main/examples/notebooks/recover_comparison.ipynb) |
| <sub>09</sub> | <sub>[`safety_monitoring`](examples/notebooks/safety_monitoring.ipynb)</sub> | <sub>Evaluate</sub> | <sub>`SpectralEntropyMonitor` catches safety drift mid-fine-tune</sub> | <sub>No GPU</sub> | [<img src="https://colab.research.google.com/assets/colab-badge.svg" height="32" alt="Open In Colab">](https://colab.research.google.com/github/Lexsi-Labs/SafeTune/blob/main/examples/notebooks/safety_monitoring.ipynb) |
| <sub>10</sub> | <sub>[`full_pipeline`](examples/notebooks/full_pipeline.ipynb)</sub> | <sub>All pillars</sub> | <sub>Measure → Diagnose → Recover → Verify → Deploy, chained end to end</sub> | <sub>GPU helps</sub> | [<img src="https://colab.research.google.com/assets/colab-badge.svg" height="32" alt="Open In Colab">](https://colab.research.google.com/github/Lexsi-Labs/SafeTune/blob/main/examples/notebooks/full_pipeline.ipynb) |

Full write-up, including which script mirrors which notebook, is in
[Notebooks](docs/examples/notebooks.md).

## Pick one per task

New here? Start with these defaults and explore the alternatives later.

| I want to… | Start with | Namespace |
|---|---|---|
| keep safety while fine-tuning | `SafeGradTrainer` | `safetune.runner.harden` |
| restore safety in a drifted model (no training) | `ReStaTrainer` | `safetune.runner.recover` |
| refuse harmful prompts at inference | `RefusalDirectionTrainer` | `safetune.runner.steer` |
| remove a capability from a model | `RMUTrainer` / `NPOTrainer` | `safetune.runner.unlearn` |
| find where safety lives | `identify_safety_neurons` | `safetune.interpret` |
| measure safety | `safetune.evaluate.evaluate()` | `safetune.evaluate` |

Each row has many alternatives — the full catalog is the
[taxonomy](docs/getting-started/taxonomy.md).

## CLI

After `pip install safetune`, the `safetune` command is available:

```bash
# Harden — train-time defence
safetune train  --model Qwen/Qwen2.5-0.5B-Instruct --algo lisa --epochs 3

# Recover — weight-space patching (no training)
safetune patch  --model ./drifted --algo resta --base ./base

# Evaluate — safety benchmarks
safetune eval   --model Qwen/Qwen2.5-0.5B-Instruct --dataset harmbench

# List all available methods
safetune list
```

Key flags for `train`:

| Flag | Default | Description |
|---|---|---|
| `--algo` | `safegrad` | Method alias (see `safetune list`) |
| `--train-dataset` | `beavertails` | `beavertails` or any HF dataset id |
| `--train-split` | `30k_train` | Split to load (e.g. `train`, `test`) |
| `--config` | — | Load all flags from a YAML file |
| `--epochs` / `--batch-size` / `--lr` | sensible defaults | Standard training knobs |

Put all flags in a YAML file and pass `--config`; explicit flags override it:

```yaml
# run.yaml
algo: lisa
model: Qwen/Qwen2.5-0.5B-Instruct
epochs: 3
train_dataset: openai/gsm8k
train_split: train
lisa_rho: 0.2          # method-specific kwargs flow straight to the trainer
```

```bash
safetune train --config run.yaml                # YAML sets defaults
safetune train --config run.yaml --epochs 5     # explicit flag wins
```

You can also add a method to the registry without touching library files:

```python
from safetune.runner._registry import register_harden
register_harden("mymethod", "MyTrainer")  # MyTrainer in safetune.runner.harden
```

Full CLI reference: [docs/user-guide/usage.md](docs/user-guide/usage.md). How to
register a method end to end: [docs/community/dev-runbook.md](docs/community/dev-runbook.md).

## How it's organized

SafeTune sorts its methods by one question: *what do you hand the method, and
when is safety enforced?* That gives two tiers. The
[taxonomy](docs/getting-started/taxonomy.md) is the single source of truth.

**Tier 1 · Interventions** — methods that *change* a model's safety. Each cell
is a catalog of independent alternatives:

| Class | You provide | Effect | Namespace |
|---|---|---|---|
| **Train-time** | base model + your fine-tuning data | `harden` — change the fine-tuning itself | `safetune.harden` |
| **Weight-space** | a finished / drifted model | `recover` lost safety, `unlearn` a capability — edit weights, no training | `safetune.recover`, `safetune.unlearn` |
| **Inference-time** | any model + steering artifacts | `steer` — wrap a frozen model, weights untouched | `safetune.steer` |

**Tier 2 · Instrumentation** — methods that *observe* safety. They support the
interventions and also stand on their own:

| Function | Effect | Namespace |
|---|---|---|
| **Diagnose** | `interpret` — find where safety lives (directions, neurons, circuits) | `safetune.interpret` |
| **Measure** | `evaluate` — red-team stressors plus benchmark/judge eval | `safetune.evaluate` |

The three intervention classes act at different points in a model's lifecycle,
so they use different usage contracts and are scored by different protocols —
checkpoint (Recover/Unlearn), paired-training (Harden), and live wrapper (Steer):

```python
from safetune.runner import recover, harden, steer, unlearn
from safetune.evaluate import evaluate

# Recover — weight-space patching, no training
trainer = recover.ReStaTrainer(drifted_model, base_model=base, aligned_model=aligned)
patched = trainer.apply()

# Harden — replaces your SFT trainer; it *is* the fine-tuning
trainer = harden.SafeGradTrainer(model, tokenizer)
trainer.train(train_dataset, safety_dataset=safety_dataset)

# Steer — inference-time, no weight changes
trainer = steer.RefusalDirectionTrainer(model, tokenizer)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

# Measure — score a model
results = evaluate(model, benchmarks=["harmbench"])
```

## The audit

"It imports and runs" is where most method collections stop. It isn't enough: a
method can execute cleanly and still be the wrong algorithm — wrong
hyperparameters, a missing step, a different loss. So every method in SafeTune
was read against its original paper and reference repository and given one of
five badges:

- **Faithful** — implements the cited paper. Safe to cite as that method.
- **Simplified** — reduced but algorithmically correct. Cite with caveats.
- **Variant** — a SafeTune heuristic, not the named algorithm. Don't cite it as one.
- **Wrong** / **Stub** — wrong algorithm, or not implemented.

Only Faithful methods should be cited as the named method from their paper;
each method's badge tells you where it stands. Per-method verdicts with
`file:line` evidence are in the
[Feature Map](docs/reference/feature-map.md); the audit's scope and the full
list of faithful methods are in [Trust & Scope](docs/community/scope.md).

## Documentation

| Doc | What it covers |
|---|---|
| [How to use these docs](docs/getting-started/how-to-read-these-docs.md) | navigation, search, audit badges — start here |
| [Getting started](docs/getting-started/index.md) | install, decision tree, 60-second quickstarts |
| [Taxonomy](docs/getting-started/taxonomy.md) | the 2-tier taxonomy (single source of truth) |
| [User guide](docs/user-guide/index.md) | per-pillar usage guides with code snippets |
| [Feature Map](docs/reference/feature-map.md) | every method with its audit badge |
| [Trust & Scope](docs/community/scope.md) | audit scope and the faithful-method list |
| [References](docs/reference/references.md) | per-method paper / venue / arXiv / repo table |
| [System design](docs/reference/system-design.md) | architecture, API contracts, dev runbook |
| [Notebooks](docs/examples/notebooks.md) | Colab notebooks for each pillar |
| [Examples](examples/) | runnable end-to-end scripts |

## Citation

If you use SafeTune in research, please cite the main paper:

```bibtex
@inproceedings{seth2026safetune,
  title     = {SafeTune: A Unified, Faithful Library for Auditing and
               Repairing Safety Drift in Fine-Tuned {LLM}s},
  author    = {Seth, Pratinav and Sadhu, Saisab and Kaushal, Anshul and
               Sankarapu, Vinay Kumar},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing: System Demonstrations},
  publisher = {Association for Computational Linguistics},
  year      = {2026},
  note      = {Pratinav Seth, Saisab Sadhu, and Anshul Kaushal contributed equally.},
}
```

## License

Lexsi Labs Source Available License (LSAL) v1.2, see [LICENSE.md](LICENSE.md).

- **Academic research and teaching** are free on MIT-like terms: use, modify,
  and redistribute with the notice intact.
- **Organizations** (companies, institutions, public bodies) must acknowledge
  their use to Lexsi Labs or obtain permission before internal evaluation,
  auditing, or use on their own models (Section 1A). Write to support@lexsi.ai.
- **Commercial use** (selling, SaaS, embedding) requires a separate commercial
  license from Lexsi Labs (support@lexsi.ai).
- **Unrepaired drifted checkpoints may not be deployed in production systems**
  (Responsible Use clause).

