# SafeTune examples

Runnable demos of SafeTune's safety interventions.

```
examples/
├── README.md
├── notebooks/          interactive Colab demos
├── quickstart/         60-second runnable demos (one per intervention pillar)
├── runner/             runner API pillar examples
├── demos/              specialized safety demos
└── safegrad_p/         supporting data for harden_quickstart
```

## Quickstart demos

Each intervention pillar has a runnable demo that works on a small model:

| Script | Pillar | What it does | Run time |
|---|---|---|---|
| `quickstart/quickstart.py` | Steer | Extract refusal direction, ablate it, measure refusal rate drop | ~30 s |
| `quickstart/recover_quickstart.py` | Recover | `ReStaTrainer` weight patching on synthetic checkpoints | ~10 s |
| `quickstart/harden_quickstart.py` | Harden | `SafeGradTrainer` gradient-surgery fine-tuning | ~1 min |
| `quickstart/unlearn_quickstart.py` | Unlearn | `GradientAscentTrainer` forget-set unlearning | ~1 min |
| `quickstart/interpret_quickstart.py` | Interpret | Safety circuit info + safety neuron identification | ~30 s |
| `quickstart/evaluate_quickstart.py` | Evaluate | Safety benchmark + red-team attack + entropy monitor | ~1 min |

```bash
python examples/quickstart/quickstart.py
python examples/quickstart/quickstart.py --model meta-llama/Llama-3.2-1B-Instruct
```

Integration tests live in `tests/integration/` at the project root.

## Runner API examples

One script per pillar using the `safetune.runner` API:

| Script | Pillar | Trainers exercised |
|---|---|---|
| `runner/harden.py` | Harden | Lisa, Vaccine, SAP, RepNoise |
| `runner/recover.py` | Recover | SafeMerge, SOMF, WiseFT |
| `runner/steer.py` | Steer | CAA, RefusalDirection, SafeSwitch, CAST |
| `runner/unlearn.py` | Unlearn | FLAT, NPO, GradDiff |

## Specialized demos

| Script | What it demonstrates |
|---|---|
| `demos/abliteration_and_defense.py` | Dual-use refusal direction: attack + defense + spectral entropy monitor |
| `demos/safety_config_quickstart.py` | `UnifiedSafetyConfig` declarative configuration (pure Python) |
| `demos/run_safety_neuron_inference.py` | Safety-neuron discovery + dynamic activation patching (mock) |
| `demos/interpret_demo.py` | Interpret pillar API walkthrough |
| `demos/evaluate_demo.py` | Evaluate pillar API walkthrough |
| `demos/full_pipeline.py` | End-to-end: measure drift → recover → verify |
| `demos/eval_model.py` | Run safety benchmarks on any HF model |
| `demos/compare_recover.py` | Compare RESTA vs LoX vs C-Θ on the same drift |

## Notebooks

Interactive Colab demos in [`notebooks/`](notebooks/):

| Notebook | Mirrors |
|---|---|
| `notebooks/steer_demo.ipynb` | `quickstart/quickstart.py` |
| `notebooks/recover_demo.ipynb` | `quickstart/recover_quickstart.py` |
| `notebooks/harden_demo.ipynb` | `quickstart/harden_quickstart.py` |
| `notebooks/unlearn_demo.ipynb` | gradient ascent unlearning |
| `notebooks/interpret_demo.ipynb` | `quickstart/interpret_quickstart.py` |
| `notebooks/evaluate_demo.ipynb` | `quickstart/evaluate_quickstart.py` |
