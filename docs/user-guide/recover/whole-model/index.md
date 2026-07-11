# Whole-model recovery

Restore safety by editing the entire model's weights. These methods treat the model as a single unit.

| Method | Mechanism |
|---|---|
| [Task arithmetic](task-arithmetic.md) | Add safety task vector `v = aligned - base` |
| [SOMF merge](somf-merge.md) | Subspace-oriented model fusion |
| [Pre-post merge](pre-post-merge.md) | Interpolate toward pre-fine-tuning checkpoint |
| [WiSE-FT](wise-ft.md) | Weight-space interpolation toward aligned model |
