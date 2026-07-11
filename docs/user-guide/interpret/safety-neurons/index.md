# Safety neurons

Two methods to identify which neurons are safety-relevant: a forward-free
weight-based score and an activation-based score computed from forward passes
on a contrast corpus.

| Method | Type | Description |
|---|---|---|
| [Weight-based](weight.md) | Forward-free | `|W_out · refusal_dir|` scoring |
| [Activation-based](activation.md) | Corpus | Harmful-vs-harmless activation contrast |
