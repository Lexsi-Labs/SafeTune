# Weight-based safety neuron identification

Scores each neuron by `|col(W_out) · refusal_dir|` — no forward passes needed.

```python
from safetune.steer import extract_refusal_direction
from safetune.interpret import identify_safety_neurons, SafetyNeuronConfig

# First extract refusal directions per layer
_, layer_id, direction_per_layer = extract_refusal_direction(
    model, tokenizer, harmful, harmless,
)

# Then identify safety neurons using those directions
report = identify_safety_neurons(
    model,
    direction_per_layer,
    config=SafetyNeuronConfig(method="weight", top_k_per_layer=16),
)

# report.per_layer: {layer_idx: [(neuron_idx, score), ...]}, ranked highest-first
top5 = report.per_layer[layer_id][:5]
print(f"layer {layer_id} top-5 (neuron_idx, score):", top5)

circuit = report.as_circuit_info()
```

## SafetyNeuronConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | `"weight"` | `"weight"` or `"activation"` |
| `top_k_per_layer` | `int` | `16` | Top-k neurons per layer |
| `target_layers` | `Optional[List[int]]` | `None` | Layers to score |
| `score_floor` | `float` | `0.0` | Minimum score threshold |
| `target_module` | `str` | `"mlp.down_proj"` | Module to inspect |
| `abs_rank` | `bool` | `True` | Rank by absolute score |

## When to use

Weight-based scoring is instant (no model forward passes). Best for quick
exploration when you have a refusal direction available.

## See also

- [Activation-based identification](activation.md) — measures activation contrast on real data; uses forward passes.
- `safety_circuit_info()` — runs both refusal-direction extraction and weight-based neuron scoring in one call.

```bibtex
@article{refusaldirection2024,
  title  = {Refusal in Language Models Is Mediated by a Single Direction},
  author = {Arditi, Andy and others},
  year   = {2024},
  note   = {NeurIPS 2024, arXiv:2406.11717},
}
```
