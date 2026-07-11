# User Guide

Usage guides for all six method groups — four intervention pillars and two
instrumentation tools. Each guide covers the input contract, a quick example,
and the catalog of methods.

<div class="st-taxchart">
  <div class="st-tc-root">
    <b>◆ SafeTune</b>
    <span>audited LLM-safety methods</span>
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
        <div class="st-tc-cell"><span class="st-tc-code">harden</span><small>Train-time · base + FT data</small><em></em></div>
        <div class="st-tc-cell"><span class="st-tc-code">recover</span><small>Weight-space · drifted model</small><em></em></div>
        <div class="st-tc-cell"><span class="st-tc-code">unlearn</span><small>Forget-set · remove capability</small><em></em></div>
        <div class="st-tc-cell"><span class="st-tc-code">steer</span><small>Inference-time · frozen model</small><em></em></div>
      </div>
    </div>
    <div class="st-tc-tier st-tc-t2">
      <div class="st-tc-head">
        <b>TIER 2 · INSTRUMENTATION</b>
        <span>OBSERVE safety</span>
      </div>
      <div class="st-tc-cells">
        <div class="st-tc-cell"><span class="st-tc-code">interpret</span><small>Diagnose · locate safety</small><em></em></div>
        <div class="st-tc-cell"><span class="st-tc-code">evaluate</span><small>Measure · red-team + eval</small><em></em></div>
      </div>
    </div>
  </div>
</div>

<div class="grid cards" markdown>

-   :material-weight-lifter:{ .lg .middle } **[Harden](harden.md)**

    ---

    Keep safety during fine-tuning. A Harden trainer *replaces* your SFT
    trainer — it IS the fine-tuning.

-   :material-wrench:{ .lg .middle } **[Recover](recover.md)**

    ---

    Restore safety in a drifted model. Weight-space editing, no training.
    Methods across 6 granularities.

-   :material-eraser:{ .lg .middle } **[Unlearn](unlearn.md)**

    ---

    Remove a capability from a finished model. Forget set + retain set.

-   :material-compass:{ .lg .middle } **[Steer](steer.md)**

    ---

    Refuse harmful prompts at inference time. Wraps a frozen model, no weight
    changes.

-   :material-magnify-scan:{ .lg .middle } **[Interpret](interpret.md)**

    ---

    Locate where safety lives — directions, neurons, circuits. Feeds Steer and
    localization-aware Recover.

-   :material-test-tube:{ .lg .middle } **[Evaluate](evaluate.md)**

    ---

    Measure safety with red-team attacks and benchmark/judge eval.

</div>

## Tips & best practices

1. **Pick one method per task.** SafeTune is a library of alternatives, not a
   pipeline. `recover`, `harden`, and `steer` solve different problems by
   different mechanisms — choose the one that matches your situation.
2. **Safety dataset**: BeaverTails works well for English LLMs; customize for
   other languages.
3. **Benchmarks**: HarmBench + XSTest cover most cases; add custom benchmarks
   as needed.
4. **Steering strength**: The strength parameter (named `strength` in CAA and
   refusal-direction steering, default `1.0`, or `multiplier` in SCANS/STA/AdaSteer,
   default `3.0`-`3.5`) controls how aggressively the model is steered. Increase for
   stronger safety enforcement; watch for fluency cost at very high values.
5. **Memory**: SafeGrad requires two forward passes (task + safety); use
   gradient accumulation to fit.
