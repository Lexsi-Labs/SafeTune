# Steer API

Inference-time steering — activation edits or decoding-time interventions that
change behavior **without touching weights**. Import from
`safetune.runner.steer`. Construct with the `model` (+ `tokenizer`), `calibrate()`
on contrastive prompt pairs, then run/evaluate the wrapped model.

```python
from safetune.runner.steer import CAATrainer

trainer = CAATrainer(model_id="...", model=model, tokenizer=tok, multiplier=20.0)
wrapped, _ = trainer.calibrate(harmful, harmless)
```

## Available trainers

Activation steering: `AdaSteerTrainer`, `AlphaSteerTrainer`, `CAATrainer`,
`CASTTrainer`, `CircuitBreakerRRTrainer`, `CircuitBreakerTrainer`,
`LinearProbeGuardTrainer`, `RRFAEnsembleTrainer`, `RefusalDirectionTrainer`,
`RepBendTrainer`, `SCANSTrainer`, `STATrainer`, `SafeSteerTrainer`,
`SafeSwitchTrainer`, `TARSteerTrainer`.

Decoding steering: `ContrastiveDecodingTrainer`, `NudgingTrainer`,
`ProxyTuningTrainer`, `SafeDecodingTrainer`.

See the [Steer guide](../../user-guide/steer.md) for the activation / decoding /
backend breakdown.

## Reference

::: safetune.runner.steer.CAATrainer
    options:
      show_source: false
      heading_level: 3

::: safetune.runner.steer.RefusalDirectionTrainer
    options:
      show_source: false
      heading_level: 3
