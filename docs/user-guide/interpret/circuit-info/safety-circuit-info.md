# safety_circuit_info()

Runs refusal-direction extraction + safety-neuron localization in one call.
Returns a `CircuitInfo`.

```python
from safetune.interpret import safety_circuit_info

circuit = safety_circuit_info(
    model, tokenizer,
    harmful_prompts=harmful, harmless_prompts=harmless,
    method="weight",
    top_k_per_layer=16,
)

print(len(circuit.safety_units.unit_ids), "safety units found")
print(circuit.safety_units.unit_ids[:5])  # e.g. ["L0.mlp.down_proj.50", ...]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | `"weight"` | Neuron scoring method |
| `top_k_per_layer` | `int` | `16` | Top-k neurons per layer |
| `target_module` | `str` | `"mlp.down_proj"` | Module to inspect |
| `target_layers` | `Optional[List[int]]` | `None` | Layers to score |
| `activation_module` | `str` | `"mlp.act_fn"` | Module for activation hooks |
| `activation_score` | `str` | `"mean_abs_diff"` | Activation score variant |

## When to use

Use `safety_circuit_info()` as the default entry point when you need a
`CircuitInfo` for downstream use in Steer or Recover. It replaces two
separate calls.
