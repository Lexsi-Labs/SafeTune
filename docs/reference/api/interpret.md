# Interpret API

Locate the parts of a model responsible for safety behavior. Import from
`safetune.interpret`. This pillar is functions + config/report dataclasses rather
than trainers.

```python
from safetune.interpret import identify_safety_neurons, safety_circuit_info

# One-call wrapper: refusal-direction extraction + weight-based neuron scan
circuit = safety_circuit_info(model, tokenizer, harmful_prompts, harmless_prompts)

# Or score neurons directly against precomputed per-layer refusal directions
report = identify_safety_neurons(model, refusal_direction_per_layer)
```

## Public surface

Functions: `identify_safety_neurons`, `safety_circuit_info`,
`eap_safety_circuit`.
Configs / reports: `SafetyNeuronConfig`, `SafetyNeuronReport`,
`EAPSafetyCircuitConfig`.

See the [Interpret guide](../../user-guide/interpret.md).

## Reference

::: safetune.interpret.identify_safety_neurons
    options:
      show_source: false
      heading_level: 3

::: safetune.interpret.safety_circuit_info
    options:
      show_source: false
      heading_level: 3

::: safetune.interpret.SafetyNeuronConfig
    options:
      show_source: false
      heading_level: 3
