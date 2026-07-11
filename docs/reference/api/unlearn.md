# Unlearn API

Remove specific harmful capabilities or knowledge. Import from
`safetune.runner.unlearn`. Construct with the `model`, then call
`unlearn(forget, retain)`.

```python
from safetune.runner.unlearn import NPOTrainer

trainer = NPOTrainer(model_id="...", model=model, num_epochs=5)
unlearned = trainer.unlearn(forget_ds, retain_ds)
```

## Available trainers

`FLATTrainer`, `GradDiffTrainer`, `GradientAscentTrainer`, `NPOTrainer`,
`RMUTrainer`, `SimDPOTrainer`.

See the [Unlearn guide](../../user-guide/unlearn.md) for when to use each.

## Reference

::: safetune.runner.unlearn.NPOTrainer
    options:
      show_source: false
      heading_level: 3

::: safetune.runner.unlearn.RMUTrainer
    options:
      show_source: false
      heading_level: 3
