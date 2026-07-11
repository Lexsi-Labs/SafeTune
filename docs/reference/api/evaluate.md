# Evaluate API

Measure safety and robustness: benchmark suites, red-team attacks, judges, and
runtime monitoring. Import from `safetune.evaluate`.

```python
from safetune.evaluate import evaluate, AbliterationAttack, TamperBenchEvaluator

results = evaluate(model, benchmarks=["harmbench"], tokenizer=tok)
```

## Public surface

Entry points: `evaluate`, `redteam`, `suite`.
Attacks: `AbliterationAttack`, `BoNAttack` (`BoNConfig`).
Judges & monitoring: `JudgeAdapter`, `SpectralEntropyMonitor`
(`SpectralMonitorConfig`), `TamperBenchEvaluator`.

See the [Evaluate guide](../../user-guide/evaluate.md).

## Reference

::: safetune.evaluate.evaluate
    options:
      show_source: false
      heading_level: 3

::: safetune.evaluate.AbliterationAttack
    options:
      show_source: false
      heading_level: 3

::: safetune.evaluate.TamperBenchEvaluator
    options:
      show_source: false
      heading_level: 3
