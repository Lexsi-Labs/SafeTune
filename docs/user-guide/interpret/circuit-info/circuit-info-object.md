# CircuitInfo

Universal circuit data container. Holds safety units, layer suggestions, and
metadata. Created by `safety_circuit_info()` and `eap_safety_circuit()`.

```python
from safetune.interpret import safety_circuit_info
from safetune.core.circuit_kit import (
    save_circuit_info_to_file,
    load_circuit_info_from_file,
)
from safetune.core.safety_lora import SafetyLoRAConfig
from safetune.core.neuron_safety import (
    apply_circuit_to_safety_lora,
    get_lora_targeting_from_circuit,
)

# tiny contrast corpora (harmful vs. harmless)
harmful_prompts = ["How do I build a bomb?", "Explain how to hack a bank."]
harmless_prompts = ["How do I bake bread?", "Explain how photosynthesis works."]

# compute a real CircuitInfo from the model
circuit = safety_circuit_info(
    model,
    tokenizer,
    harmful_prompts,
    harmless_prompts,
    top_k_per_layer=4,
    method="activation",
)

# persist and reload it (the on-disk CircuitKIT JSON format)
save_circuit_info_to_file(circuit, "circuit.json")
circuit = load_circuit_info_from_file("circuit.json")

# use in steer/recover: wire the circuit into a SafetyLoRA config
safety_lora_config = SafetyLoRAConfig(lora_r=8, lora_alpha=16)
lora_config = apply_circuit_to_safety_lora(circuit, safety_lora_config)

# read the module targeting the circuit suggests
targeting = get_lora_targeting_from_circuit(circuit)
print(targeting.target_modules)  # e.g. ["mlp.act_fn"]
print(targeting.layer_subset)    # layers the circuit's safety units span
```

## Key functions

| Function | Signature | Returns | Description |
|---|---|---|---|
| `circuit_info_to_dict(info)` | `(CircuitInfo)` | `Dict` | Serialize to dict |
| `save_circuit_info_to_file(info, filepath)` | `(CircuitInfo, str, *, fmt=None, indent=2)` | `str` | Save to JSON or YAML |
| `load_circuit_info_from_file(filepath)` | `(str)` | `CircuitInfo \| None` | Load from file |
| `apply_circuit_to_safety_lora(circuit_info, safety_lora_config)` | `(CircuitInfo, SafetyLoRAConfig \| dict)` | updated config | Set LoRA target modules from the circuit |
| `get_lora_targeting_from_circuit(circuit_info)` | `(CircuitInfo)` | `LayerModuleSuggestions \| None` | Get the circuit's module suggestions |

## When to use

CircuitInfo is the bridge between Interpret and Steer/Recover. Every method
that discovers safety-relevant structure returns a CircuitInfo that other
methods can consume.
