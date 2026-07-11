# Harden API

Train-time defenses. Import from `safetune.runner.harden`. Each trainer takes
`model, tokenizer, **hyperparams` and exposes `train(train_dataset, out_dir=...)`.

```python
from safetune.runner.harden import SafeGradTrainer

trainer = SafeGradTrainer(model, tokenizer)
path = trainer.train(train_dataset, out_dir="./hardened")
```

## Available trainers

`AntibodyTrainer`, `AsFTTrainer`, `BoosterTrainer`, `CSTTrainer`, `CTRAPTrainer`,
`ConstrainedSFTTrainer`, `DOORTrainer`, `DeRTaTrainer`, `DeepRefusalTrainer`,
`LisaTrainer`, `LoXHardenTrainer`, `LookAheadTrainer`, `MARTTrainer`,
`PlainSFTTrainer`, `RepNoiseTrainer`, `SAPTrainer`, `SEALTrainer`, `SEAMTrainer`,
`SPPFTTrainer`, `STARDSSTrainer`, `SaLoRATrainer`, `SafeGradTrainer`,
`SurgeryTrainer`, `TARTrainer`, `TVaccineTrainer`, `VaccineTrainer`.

See the [Harden guide](../../user-guide/harden.md) for method selection. A few methods
(`CSTTrainer`, `MARTTrainer`, `DeepRefusalTrainer`, `AntibodyTrainer`) are
programmatic-only — see the [CLI Reference](../cli.md).

## Reference

::: safetune.runner.harden.SafeGradTrainer
    options:
      show_source: false
      heading_level: 3
      members_order: source

::: safetune.runner.harden.LisaTrainer
    options:
      show_source: false
      heading_level: 3

::: safetune.runner.harden.SAPTrainer
    options:
      show_source: false
      heading_level: 3
