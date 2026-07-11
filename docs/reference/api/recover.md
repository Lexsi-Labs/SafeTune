# Recover API

Weight-space repair of a model whose safety has drifted. Import from
`safetune.runner.recover`. Recover trainers are training-free: construct with the
drifted `model` (plus `base_model` / `aligned_model` references where the method
needs them) and call `apply()`.

```python
from safetune.runner.recover import ReStaTrainer

trainer = ReStaTrainer(model=drifted, base_model=base, aligned_model=aligned,
                       alpha=0.5, dare=True, dare_seed=0)
repaired = trainer.apply()
```

## Available trainers

`AAQTrainer`, `AntidoteTrainer`, `AntidoteV2Trainer`, `CThetaTrainer`,
`GradSelectiveRecoverTrainer`, `LSSFTrainer`, `LoXTrainer`, `MSCPTrainer`,
`NLSRTrainer`, `OneShotSafetyPatchTrainer`, `PKETrainer`, `PrePostMergeTrainer`,
`QReSafeTrainer`, `ReStaTrainer`, `RepNoiseRecoverTrainer`, `SCRUBTrainer`,
`SOMFTrainer`, `SafeDeltaTrainer`, `SafeLoRATrainer`, `SafeMergeTrainer`,
`SafeReActTrainer`, `SafetyVectorRestoreTrainer`, `TaskArithmeticTrainer`,
`WiseFTTrainer`.

See the [Recover guide](../../user-guide/recover.md) for the method taxonomy
(whole-model / low-rank / layer / neuron / saliency / circuit-guided).

## Reference

::: safetune.runner.recover.ReStaTrainer
    options:
      show_source: false
      heading_level: 3

::: safetune.runner.recover.TaskArithmeticTrainer
    options:
      show_source: false
      heading_level: 3

::: safetune.runner.recover.SafeLoRATrainer
    options:
      show_source: false
      heading_level: 3
