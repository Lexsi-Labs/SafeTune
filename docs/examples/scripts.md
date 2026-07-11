# Python Scripts

The scripts under `examples/` are small, self-contained, and grouped by
purpose. Most run on CPU with `Qwen/Qwen2.5-0.5B-Instruct` and take no setup;
the `runner/` and pipeline scripts need your own drifted / base / aligned
checkpoints. Run every command from the repo root.

---

## `quickstart/` — 60-second onramps

One runnable demo per pillar. All default to `Qwen/Qwen2.5-0.5B-Instruct` and
accept `--model` and `--device`.

### quickstart.py

Walks the inference-time STEER path end to end: extract a refusal direction
from contrast prompts, ablate it live, and print the change — no training.

```bash
python examples/quickstart/quickstart.py
```

### harden_quickstart.py

Defends a model *during* fine-tuning with `SafeGradTrainer`, which projects the
task gradient off the alignment gradient so contaminated data does not erode
safety. Trains a few steps on tiny synthetic data.

```bash
python examples/quickstart/harden_quickstart.py
```

### recover_quickstart.py

Restores safety on a drifted model with `ReStaTrainer` weight patching
(`θ_target += α·(θ_aligned − θ_base)`), synthesising the three checkpoints from
one small model — no training.

```bash
python examples/quickstart/recover_quickstart.py
```

### unlearn_quickstart.py

Removes a capability from a finished model with `GradientAscentTrainer`, which
maximises loss on a harmful forget set while preserving a retain set.

```bash
python examples/quickstart/unlearn_quickstart.py
```

### interpret_quickstart.py

Locates safety circuits: extracts safety-relevant neurons and builds a
`CircuitInfo` from contrastive harmful/harmless prompts.

```bash
python examples/quickstart/interpret_quickstart.py
```

### evaluate_quickstart.py

Measures a model's safety alignment — runs a safety benchmark, a red-team
attack, and spectral entropy monitoring.

```bash
python examples/quickstart/evaluate_quickstart.py
```

---

## `demos/` — focused feature tours

Longer walkthroughs of individual APIs. Several print the API surface rather
than train, so they run instantly.

### abliteration_and_defense.py

Demonstrates the dual-use refusal-direction primitive (Arditi et al., 2024):
the same direction that breaks safety via ablation can strengthen it via
steering, paired with the spectral entropy monitor.

```bash
python examples/demos/abliteration_and_defense.py
```

### compare_recover.py

Compares recovery methods (RESTA, LoX, C-Θ) on the same drifted model to show
the safety-vs-capability trade-off. Needs `SAFETUNE_DRIFT_MODEL`,
`SAFETUNE_BASE_MODEL`, and `SAFETUNE_ALIGNED_MODEL`.

```bash
python examples/demos/compare_recover.py
```

### eval_model.py

Runs safety benchmarks on any Hugging Face model via `safetune.evaluate`.

```bash
python examples/demos/eval_model.py --model Qwen/Qwen2.5-0.5B-Instruct
```

### evaluate_demo.py

Tours the Measure pillar's three functions: red-team attacks (abliteration,
BoN), benchmark scoring through a judge, and the spectral entropy monitor.

```bash
python examples/demos/evaluate_demo.py
```

### full_pipeline.py

The canonical end-to-end workflow — measure drift → recover safety → verify —
for restoring a fine-tuned model. Needs `SAFETUNE_DRIFT_MODEL`,
`SAFETUNE_BASE_MODEL`, and `SAFETUNE_ALIGNED_MODEL`.

```bash
python examples/demos/full_pipeline.py
```

### interpret_demo.py

Tours the Diagnose pillar's entry points: `identify_safety_neurons`,
`safety_circuit_info`, and `eap_safety_circuit` (EAP / EAP-IG).

```bash
python examples/demos/interpret_demo.py
```

### run_safety_neuron_inference.py

A mock illustration of the SafetyNeuron inference pipeline — dynamic activation
patching and the LLM Safeguard predictor — using lightweight stand-ins so it
runs instantly with no GPU or download.

```bash
python examples/demos/run_safety_neuron_inference.py
```

### safety_config_quickstart.py

Demonstrates the declarative `UnifiedSafetyConfig` that toggles every safety
module across training, inference, guardrails, and evaluation. Pure Python — no
model or GPU.

```bash
python examples/demos/safety_config_quickstart.py
```

---

## `runner/` — uniform Trainer interface per pillar

Each script sweeps several methods in a pillar through the runner API. They
require your own checkpoints (set via env vars or edit the `_MODEL_ID` at the
top); for a runnable version use the matching `quickstart/` demo instead.

### harden.py

Sweeps LISA, Vaccine, SAP, and RepNoise harden trainers: `train` → `evaluate`.
Needs `SAFETUNE_MODEL` and `SAFETUNE_DOMAIN`.

```bash
python examples/runner/harden.py
```

### recover.py

Sweeps SafeMerge, SOMF, and WiseFT recover trainers: `apply` → `evaluate`.
Needs `SAFETUNE_DRIFT_MODEL`, `SAFETUNE_BASE_MODEL`, `SAFETUNE_ALIGNED_MODEL`,
and `SAFETUNE_DOMAIN`.

```bash
python examples/runner/recover.py
```

### steer.py

Sweeps CAA, RefusalDirection, SafeSwitch, and CAST steer trainers: `calibrate`
→ `evaluate`. Needs `SAFETUNE_MODEL` and `SAFETUNE_DOMAIN`.

```bash
python examples/runner/steer.py
```

### unlearn.py

Sweeps FLAT, NPO, and GradDiff unlearn trainers over a forget / retain split.

```bash
python examples/runner/unlearn.py
```

---

## `case_studies/`

### single_direction_jailbreak.py

A fully reproducible, judge-free red-team demonstration (runs in ~2 minutes on
CPU): measures an aligned model's refusal rate, runs `AbliterationAttack` to
ablate the single refusal direction, and re-measures. A large drop shows safety
leans on one cheaply-removable direction.

```bash
python examples/case_studies/single_direction_jailbreak.py
```

See the write-up in
[Single-direction jailbreak](../examples/single-direction-jailbreak.md).
