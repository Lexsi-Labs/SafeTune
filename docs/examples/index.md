# Examples

SafeTune ships runnable examples under `examples/`. They come in three forms:

- **[Python Scripts](scripts.md)** — small, self-contained `.py` files grouped
  by purpose (`quickstart/`, `demos/`, `runner/`, `case_studies/`). Most run on
  CPU with `Qwen/Qwen2.5-0.5B-Instruct` and are the recommended starting point
  for integration and CI.
- **[Notebooks](notebooks.md)** — 10 Colab-ready `.ipynb` walkthroughs, one per
  pillar (plus comparison and full-pipeline notebooks) with narration and
  plots.
- **Case study** — `examples/case_studies/single_direction_jailbreak.py`, a
  fully reproducible, judge-free red-team demonstration that refusal in an
  aligned model rides on a single direction. Written up in
  [Single-direction jailbreak](../examples/single-direction-jailbreak.md).

## By pillar

SafeTune spans six pillars. Pick the example that matches the mechanism you
need:

| Pillar | What it does | Quickstart script | Demo / notebook |
|---|---|---|---|
| **Harden** | Train-time defence (replaces your SFT trainer) | `quickstart/harden_quickstart.py` | `runner/harden.py`, `harden_demo.ipynb` |
| **Recover** | Weight-space repair of a drifted model — no training | `quickstart/recover_quickstart.py` | `demos/compare_recover.py`, `runner/recover.py`, `recover_demo.ipynb`, `recover_comparison.ipynb` |
| **Unlearn** | Remove a capability via forget-set training | `quickstart/unlearn_quickstart.py` | `runner/unlearn.py`, `unlearn_demo.ipynb` |
| **Steer** | Inference-time steering — no weight edits | `quickstart/quickstart.py` | `demos/abliteration_and_defense.py`, `runner/steer.py`, `steer_demo.ipynb`, `steer_comparison.ipynb` |
| **Interpret** | Locate safety circuits and neurons | `quickstart/interpret_quickstart.py` | `demos/interpret_demo.py`, `demos/run_safety_neuron_inference.py`, `interpret_demo.ipynb` |
| **Evaluate** | Measure safety: benchmarks, red-team, drift monitor | `quickstart/evaluate_quickstart.py` | `demos/eval_model.py`, `demos/evaluate_demo.py`, `evaluate_demo.ipynb`, `safety_monitoring.ipynb` |

The end-to-end `demos/full_pipeline.py` and `full_pipeline.ipynb` chain
several pillars (Measure → Diagnose → Recover → Verify → Deploy).

## Running them

```bash
# From the repo root, any script:
python examples/quickstart/quickstart.py

# Override the default model:
python examples/quickstart/quickstart.py --model meta-llama/Llama-3.2-1B-Instruct
```

See [Python Scripts](scripts.md) for the full catalogue and per-script run
commands, or [Notebooks](notebooks.md) for the Colab-ready walkthroughs.
