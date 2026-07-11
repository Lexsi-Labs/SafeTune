# Activation-based safety neuron identification

Scores neurons by harmful-vs-harmless activation contrast across a corpus.
Three score variants: `mean_abs_diff`, `tstat`, `mean_diff`.

```python
from safetune.interpret import identify_safety_neurons, SafetyNeuronConfig

report = identify_safety_neurons(
    model,
    None,  # refusal_direction_per_layer not needed for activation mode
    tokenizer=tokenizer,
    harmful_prompts=harmful, harmless_prompts=harmless,
    config=SafetyNeuronConfig(method="activation", activation_score="tstat"),
)

# report.per_layer: {layer_idx: [(neuron_idx, tstat_score), ...]}, ranked highest-first
top5 = report.per_layer[0][:5]
print("layer 0 top-5 (neuron_idx, tstat):", top5)

circuit = report.as_circuit_info()
```

## SafetyNeuronConfig (extra fields for activation method)

| Field | Type | Default | Description |
|---|---|---|---|
| `activation_module` | `str` | `"mlp.act_fn"` | Module to hook for activations |
| `activation_score` | `str` | `"mean_abs_diff"` | `"mean_abs_diff"`, `"tstat"`, or `"mean_diff"` |
| `activation_batch_size` | `int` | `8` | Forward-pass batch size |
| `activation_max_tokens` | `int` | `64` | Max tokens per prompt |

## When to use

Activation-based scoring measures each neuron's activation contrast on real
harmful-vs-harmless prompts, rather than the weight-space proxy used by the
weight method. Use it when you can afford forward passes on a corpus. Note
that the implemented contrast (mean activation magnitude on harmful minus
harmless prompts, optionally standardized) is a localization heuristic on a
single model; it is not identical to the metrics in the papers cited below.

## See also

- [Weight-based identification](weight.md) — instant, no forward passes; good for quick exploration.
- `safety_circuit_info()` — runs refusal-direction extraction + weight-based scoring in one call.

```bibtex
@article{wei2024pruning,
  title  = {Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications},
  author = {Wei, Boyi and Huang, Kaixuan and Huang, Yangsibo and others},
  year   = {2024},
  note   = {ICML 2024, arXiv:2402.05162},
}

@article{chen2024safetyneurons,
  title  = {Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons},
  author = {Chen, Jianhui and Wang, Xiaozhi and Yao, Zijun and others},
  year   = {2024},
  note   = {arXiv:2406.14144},
}
```
