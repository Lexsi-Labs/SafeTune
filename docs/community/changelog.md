# Changelog

## v1.0.0 — Initial public release

SafeTune ships a broad roster of LLM-safety methods, organized into a 2-tier taxonomy and
faithfulness-audited against their cited papers.

### Methods included

| Pillar | Methods | What it does |
|---|---|---|
| **Harden** | 27 | Training-time defenses — keep safety intact through fine-tuning |
| **Recover** | 26 | Training-free weight-space recovery for drifted models |
| **Unlearn** | 6 | Capability removal via forget-set training |
| **Steer** | 19 | Inference-time intervention without weight changes |
| **Interpret** | 6 | Refusal-direction extraction, safety-neuron localization, circuit discovery |
| **Evaluate** | 24 | Red-team attacks and benchmark scoring |

### Configuration and extensibility

- **`SafeTuneConfig`** — declarative YAML config for CLI runs
  (`SafeTuneConfig.from_yaml`), plus the `--config` CLI flag.
- **`--train-dataset` / `--train-split`** CLI flags — training data fully
  configurable.
- **`safetune.runner._registry`** — centralised algo registry;
  `register_harden()` / `register_recover()` / `register_unlearn()` let
  third-party code extend the method menu at runtime.

### Audit summary

Every method was audited against its cited paper and carries a badge: most are
Faithful, a few are Simplified-correct or Variant (they run correctly but must
not be cited as the named method). See the
[Feature Map](../reference/feature-map.md) for per-method verdicts and
[Scope & Limitations](../community/scope.md) for what each verdict means.

### Getting started

```bash
pip install safetune
```

See [Getting started](../getting-started/index.md) for a 60-second quickstart,
[User Guide](../user-guide/index.md) for per-pillar usage, and
[Notebooks](../examples/notebooks.md) for Colab notebooks.
