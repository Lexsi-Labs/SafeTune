# API Reference

SafeTune exposes two layers for the same methods:

| Layer | Import | Best for |
|-------|--------|----------|
| **Runner** (recommended) | `from safetune.runner.<pillar> import <Trainer>` | One-line, config-driven training/eval with checkpointing and metric rollup. |
| **Interventions** (low-level) | `from safetune.<pillar> import ...` | Composing a method into your own training loop or research code. |

Every method is a self-contained class or function — you **pick one per task**, you do not chain them. The six pillars:

| Pillar | Module | What it does |
|--------|--------|--------------|
| [Harden](harden.md) | `safetune.runner.harden` | Train-time defenses that keep a model safe *while* it is fine-tuned. |
| [Recover](recover.md) | `safetune.runner.recover` | Weight-space repair of a model whose safety has already drifted. |
| [Unlearn](unlearn.md) | `safetune.runner.unlearn` | Remove specific harmful capabilities/knowledge. |
| [Steer](steer.md) | `safetune.runner.steer` | Inference-time activation / decoding steering (no weight changes). |
| [Interpret](interpret.md) | `safetune.interpret` | Locate safety neurons and safety circuits. |
| [Evaluate](evaluate.md) | `safetune.evaluate` | Safety benchmarks, red-team attacks, judges, monitoring. |

Shared [configuration](config.md) is described on its own page.

## Import cheat-sheet

```python
# Runner layer — the common path
from safetune.runner.harden  import SafeGradTrainer
from safetune.runner.recover import ReStaTrainer
from safetune.runner.unlearn import NPOTrainer
from safetune.runner.steer   import CAATrainer

# Interpret / Evaluate are functions + configs
from safetune.interpret import identify_safety_neurons, safety_circuit_info
from safetune.evaluate  import evaluate, AbliterationAttack, TamperBenchEvaluator
```

Each trainer takes a `model` (and, where relevant, a `tokenizer`) plus method-specific
keyword arguments, then exposes `train(...)` / `apply(...)` / `evaluate(...)` depending
on the pillar. See each pillar page for the exact surface.
