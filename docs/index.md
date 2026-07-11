---
title: SafeTune
hide:
  - navigation
  - toc
---

<div class="st-hero" markdown>

<img class="st-hero-mark" src="assets/compass.svg" alt="">

<h1 class="st-hero-title">Fine-tuning breaks safety. <em>SafeTune fixes it.</em></h1>

<p class="st-tagline">
You fine-tuned an aligned model and it quietly stopped refusing harmful
requests. SafeTune gives you ~100 methods to fix that — harden the
fine-tuning, recover drifted weights, steer at inference, or unlearn a
capability — each audited against its source paper, so you know what you're
actually running.
</p>

[Get started](getting-started/index.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/Lexsi-Labs/SafeTune){ .md-button }

<p class="st-chips">
<span>v1.0.0</span>
<span>MIT</span>
<span>Python 3.12+</span>
<span>~100 methods</span>
<span>every method audited</span>
</p>

</div>

## Pick one method per task — not a pipeline

SafeTune is a **library of alternatives**, not a sequence. Each task has many
independent methods that solve it by *different mechanisms*. You pick one. You
don't chain them.

<div class="st-taxchart">
  <div class="st-tc-root">
    <b>◆ SafeTune</b>
    <span>~100 audited LLM-safety methods</span>
  </div>
  <div class="st-tc-conn">
    <div class="st-tc-conn-branch st-tc-t1-branch">
      <span class="st-tc-conn-label">Tier 1 · Interventions</span>
      <span class="st-tc-conn-arrow">↓</span>
    </div>
    <div class="st-tc-conn-branch st-tc-t2-branch">
      <span class="st-tc-conn-label">Tier 2 · Instrumentation</span>
      <span class="st-tc-conn-arrow">↓</span>
    </div>
  </div>
  <div class="st-tc-branches">
    <div class="st-tc-tier st-tc-t1">
      <div class="st-tc-head">
        <b>TIER 1 · INTERVENTIONS</b>
        <span>CHANGE model safety</span>
      </div>
      <div class="st-tc-cells">
        <div class="st-tc-cell"><span class="st-tc-code">harden</span><small>Train-time · base + FT data</small><em>27 methods</em></div>
        <div class="st-tc-cell"><span class="st-tc-code">recover</span><small>Weight-space · drifted model</small><em>26 methods</em></div>
        <div class="st-tc-cell"><span class="st-tc-code">unlearn</span><small>Forget-set · remove capability</small><em>6 methods</em></div>
        <div class="st-tc-cell"><span class="st-tc-code">steer</span><small>Inference-time · frozen model</small><em>19 methods</em></div>
      </div>
    </div>
    <div class="st-tc-tier st-tc-t2">
      <div class="st-tc-head">
        <b>TIER 2 · INSTRUMENTATION</b>
        <span>OBSERVE safety</span>
      </div>
      <div class="st-tc-cells">
        <div class="st-tc-cell"><span class="st-tc-code">interpret</span><small>Diagnose · locate safety</small><em>6 methods</em></div>
        <div class="st-tc-cell"><span class="st-tc-code">evaluate</span><small>Measure · red-team + eval</small><em>24 methods</em></div>
      </div>
    </div>
  </div>
</div>

| I want to… | Try | Pillar |
|---|---|---|
| keep safety **during** fine-tuning | `harden.SafeGradTrainer` | `safetune.runner.harden` |
| restore safety **after** drift (no training) | `recover.ReStaTrainer` | `safetune.runner.recover` |
| refuse harmful prompts **at inference** | `steer.RefusalDirectionTrainer` | `safetune.runner.steer` |
| **remove** a capability | `unlearn.RMUTrainer` | `safetune.runner.unlearn` |
| **locate** where safety lives | `safety_circuit_info` | `safetune.interpret` |
| **measure** safety | `evaluate()` | `safetune.evaluate` |

New here? Read [How to use these docs](getting-started/how-to-read-these-docs.md) for
navigation tips and a guide to the audit badge system.

## See it in 60 seconds

=== "Steer"

    ```python
    from safetune.runner import steer

    # harmful_prompts / harmless_prompts — ~20 contrast examples each (list of str)
    trainer = steer.RefusalDirectionTrainer(model, tokenizer)
    wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)
    ```

    Run it: `python examples/quickstart/quickstart.py`

=== "Recover"

    ```python
    from safetune.runner import recover

    # drifted_model  — your fine-tuned checkpoint (AutoModelForCausalLM)
    # base_model     — the original pre-trained model
    # aligned_model  — the safety-aligned reference (e.g. the RLHF checkpoint)
    trainer = recover.ReStaTrainer(
        drifted_model, base_model=base_model, aligned_model=aligned_model
    )
    patched = trainer.apply()
    ```

    Run it: `python examples/quickstart/recover_quickstart.py`

=== "Harden"

    ```python
    from safetune.runner import harden

    # train_dataset  — your domain fine-tuning data (any HuggingFace Dataset)
    # safety_dataset — a safety corpus; use safetune.data.load_beavertails() or your own
    trainer = harden.SafeGradTrainer(model, tokenizer)
    trainer.train(train_dataset, safety_dataset=safety_dataset)
    ```

    Run it: `python examples/quickstart/harden_quickstart.py`

=== "Unlearn"

    ```python
    from safetune.runner import unlearn

    # forget_batches — DataLoader of prompts covering the capability to erase
    # retain_batches — DataLoader of general-purpose prompts to preserve the rest
    trainer = unlearn.RMUTrainer(model)
    trainer.unlearn(forget=forget_batches, retain=retain_batches)
    ```

    Run it: `python examples/quickstart/unlearn_quickstart.py`

=== "Interpret"

    ```python
    from safetune.interpret import safety_circuit_info

    # harmful / harmless — contrast prompt lists; same format as Steer calibration
    circuit = safety_circuit_info(
        model, tokenizer,
        harmful_prompts=harmful, harmless_prompts=harmless,
    )
    ```

    Run it: `python examples/quickstart/interpret_quickstart.py`

=== "Evaluate"

    ```python
    from safetune.evaluate import evaluate

    # "harmbench" / "xstest" — benchmark suites downloaded automatically on first run
    # judge="wildguard"       — WildGuard classifier used to score model responses
    results = evaluate(model, tokenizer=tok, benchmarks=["harmbench", "xstest"], judge="wildguard")
    print(results["harmbench"]["asr"])
    ```

    Run it: `python examples/quickstart/evaluate_quickstart.py`

## Four intervention pillars

Methods that **change** a model's safety — train-time, weight-space, or inference-time.

<div class="grid cards st-4col" markdown>

-   :material-weight-lifter:{ .lg .middle } **Harden**

    ---

    Stop safety from breaking *during* fine-tuning. Trainers replace your SFT
    loop — gradient surgery, data alternation, representation perturbation.
    26 methods · 8 families.

    :octicons-arrow-right-24: [Harden guide](user-guide/harden.md)

-   :material-wrench:{ .lg .middle } **Recover**

    ---

    Safety already broke? Patch the weights directly — no retraining.
    Whole-model arithmetic to neuron-level surgery.
    24 methods · 6 granularities.

    :octicons-arrow-right-24: [Recover guide](user-guide/recover.md)

-   :material-compass:{ .lg .middle } **Steer**

    ---

    Don't touch weights. Wrap any model with a refusal direction or logits
    processor — active only while installed. 19 methods.
    Optional vLLM backend for faster batched throughput.

    :octicons-arrow-right-24: [Steer guide](user-guide/steer.md)

-   :material-eraser:{ .lg .middle } **Unlearn**

    ---

    Make a model forget a specific skill. Trains on a forget set + retain set
    to erase knowledge while preserving the rest.
    6 methods.

    :octicons-arrow-right-24: [Unlearn guide](user-guide/unlearn.md)

</div>

## Two instrumentation tools

Methods that **observe** safety — diagnose where it lives and measure whether it holds.

<div class="grid cards" markdown>

-   :material-magnify-scan:{ .lg .middle } **Interpret — locate where safety lives**

    ---

    Find refusal directions, safety neurons, and circuits inside a model.
    The artifacts feed Steer and circuit-guided Recover — the same finding
    has three uses: steer with it, mask a weight edit, or report it. 6 methods.

    **`identify_safety_neurons`** · **`safety_circuit_info`** · **`eap_safety_circuit`**

    :octicons-arrow-right-24: [Interpret guide](user-guide/interpret.md)

-   :material-test-tube:{ .lg .middle } **Evaluate — measure what happened**

    ---

    Red-team attacks and benchmark/judge evaluation. HarmBench, XSTest,
    AdvBench, WildJailbreak. WildGuard and LlamaGuard-3 judges. One call.
    24 methods.

    **`evaluate(model, benchmarks=["harmbench"])`** · **`BoNAttack`**

    :octicons-arrow-right-24: [Evaluate guide](user-guide/evaluate.md)

</div>

## How teams use SafeTune

Four common situations and which method fits each. Code snippets use placeholder variable names — swap in your own model ID and dataset.

=== "Drift after fine-tuning"

    **Situation:** You fine-tuned an aligned model on proprietary data. It now
    complies with harmful requests the base model refused. You need safety back
    without retraining from scratch.

    ```python
    from safetune.runner import recover
    from safetune.evaluate import evaluate

    # drifted_model / base_model / aligned_model — your AutoModelForCausalLM objects
    # "harmbench" — HarmBench red-team suite, downloaded automatically on first run
    # judge="wildguard" — WildGuard classifier used to score model responses

    # 1. Measure how bad the drift is
    before = evaluate(drifted_model, tokenizer=tokenizer, benchmarks=["harmbench"], judge="wildguard")
    print(f"Refusal rate: {before['harmbench']['refusal_rate']:.0%}")  # e.g. 43%

    # 2. Patch weights directly — no training, ~30 s on a single GPU
    trainer = recover.ReStaTrainer(
        drifted_model, base_model=base_model, aligned_model=aligned_model, alpha=0.5
    )
    patched = trainer.apply()

    # 3. Confirm recovery
    after = evaluate(patched, tokenizer=tokenizer, benchmarks=["harmbench"], judge="wildguard")
    print(f"Refusal rate: {after['harmbench']['refusal_rate']:.0%}")   # e.g. 89%
    ```

    26 recovery methods from whole-model arithmetic to neuron-level surgery.
    [:octicons-arrow-right-24: Recover guide](user-guide/recover.md)

=== "Compliance-critical deployment"

    **Situation:** Fine-tuning a model for a regulated application (banking,
    healthcare, legal). Safety must hold through the training run — you can't
    patch it afterward.

    ```python
    from safetune.runner import harden

    # train_dataset  — your domain data (HuggingFace Dataset)
    # safety_dataset — safety corpus; use safetune.data.load_beavertails() or your own

    # Drop-in replacement for your SFT trainer
    trainer = harden.SafeGradTrainer(
        model, tokenizer,
        rho=0.05,           # gradient surgery strength
        kl_temperature=1.0, # KL alignment temperature
    )
    trainer.train(train_dataset, safety_dataset=safety_dataset)
    # Saves a fine-tuned checkpoint with safety preserved
    ```

    Safety-constrained training. The model learns your domain without forgetting
    refusals. [:octicons-arrow-right-24: 27 harden methods](user-guide/harden.md)

=== "Production steer — no retraining"

    **Situation:** Live production model. You cannot retrain. You need to enforce
    refusals on a new harm category within hours, not weeks.

    ```python
    from safetune.runner import steer

    # harmful_prompts / harmless_prompts — ~20 contrast examples each (list of str)
    # inputs — standard tokenizer output: {"input_ids": ..., "attention_mask": ...}

    # Calibrate on 20–30 contrastive examples (harmful / harmless)
    trainer = steer.RefusalDirectionTrainer(model, tokenizer)
    wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

    wrapped.install()  # activate the steering hooks

    # generate through the wrapped model while the hooks are active
    output = wrapped.model.generate(**inputs)

    # Revert without reloading the model
    wrapped.remove()
    ```

    Zero retraining. Reversible. Optional vLLM backend for higher throughput.
    [:octicons-arrow-right-24: 19 steer methods](user-guide/steer.md)

=== "Erase a capability"

    **Situation:** A deployed model knows something it shouldn't — a capability
    you need to remove (CBRN knowledge, PII memorisation, a deprecated skill).
    Standard fine-tuning doesn't forget cleanly.

    ```python
    from safetune.runner import unlearn

    trainer = unlearn.RMUTrainer(model)
    trainer.unlearn(
        forget=forget_batches,   # what to erase
        retain=retain_batches,   # what to keep
    )
    # Returns a checkpoint with just that capability removed
    ```

    Trains only on the forget/retain split — not a full retrain.
    [:octicons-arrow-right-24: 6 unlearn methods](user-guide/unlearn.md)

## Where next

<div class="st-wide" markdown>

| | |
|---|---|
| **[Getting started](getting-started/index.md)** | Install, quickstart, the taxonomy |
| **[Guides](user-guide/index.md)** | Per-pillar usage guides with code |
| **[Usage](user-guide/usage.md)** | CLI, YAML config, and Python API — three ways to run any method |
| **[Feature Map](reference/feature-map.md)** | Per-method audit badges — faithful, variant, simplified |
| **[Notebooks](examples/notebooks.md)** | Colab notebooks for each pillar |
| **[References](reference/references.md)** | Full paper table, eval protocols |

</div>

---

## Cite

```bibtex
@misc{seth2026safetune,
  title  = {SafeTune: A Unified Library for Preserving and Restoring
            Safety in Fine-Tuned {LLM}s},
  author = {Seth, Pratinav and Kaushal, Anshul and Sadhu, Saisab and
            Sankarapu, Vinay Kumar},
  year   = {2026},
  note   = {Pratinav Seth, Anshul Kaushal, and Saisab Sadhu contributed equally.},
}
```

---

<div class="st-lexsi-footer" markdown>
<a href="https://www.lexsi.ai">
  <img class="st-lexsi-on-light" src="assets/lexsi-logo-dark.png" alt="Lexsi Labs" width="240">
  <img class="st-lexsi-on-dark" src="assets/lexsi-logo-white.png" alt="Lexsi Labs" width="240">
</a>
<p><a href="https://www.lexsi.ai">https://www.lexsi.ai</a></p>
<p>Mumbai 🇮🇳 · Paris 🇫🇷 · London 🇬🇧</p>
</div>
