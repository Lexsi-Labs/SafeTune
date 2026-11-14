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
print(targeting.target_modules)  # e.g. ["mlp.down_proj"]
print(targeting.layer_subset)    # layers the circuit's safety units span
```

## Key functions

| Function | Signature | Returns | Description |
|---|---|---|---|
| `circuit_info_to_dict(info)` | `(CircuitInfo)` | `Dict` | Serialize to dict |
| `save_circuit_info_to_file(info, filepath)` | `(CircuitInfo, str, *, fmt=None, indent=2)` | `str` | Save to JSON or YAML |
| `load_circuit_info_from_file(filepath)` | `(str, *, top_fraction=0.2, layer_fraction=None)` | `CircuitInfo \| None` | Load from file |
| `circuit_info_from_node_scores(node_scores)` | `(dict, *, top_fraction=0.2, layer_fraction=None, ...)` | `CircuitInfo` | Build from CircuitKIT `node_scores` |
| `layer_mass(node_scores)` | `(dict)` | `dict[int, float]` | Total absolute node score per layer |
| `top_layers(node_scores, layer_fraction)` | `(dict, float)` | `list[int]` | The top fraction of layers by that total |
| `apply_circuit_to_safety_lora(circuit_info, safety_lora_config)` | `(CircuitInfo, SafetyLoRAConfig \| dict)` | updated config | Set LoRA target modules from the circuit |
| `get_lora_targeting_from_circuit(circuit_info)` | `(CircuitInfo)` | `LayerModuleSuggestions \| None` | Get the circuit's module suggestions |

## Restricting a circuit to its strongest layers

By default a CircuitKIT score file selects every layer that holds a top node. Top nodes
are spread through a model, so this can be nearly every layer. Pass `layer_fraction` to
rank layers by total absolute node score instead:

```python
circuit = load_circuit_info_from_file("x_scores.json", layer_fraction=0.2)
print(circuit.layer_suggestions.layer_subset)  # the strongest 20% of layers
```

`layer_fraction` needs `node_scores` in the file, and it replaces any `safety_units` and
`layer_suggestions` there. `top_fraction` then picks among the nodes in those layers.

## When to use

CircuitInfo is the bridge between Interpret and Steer/Recover. Every method
that discovers safety-relevant structure returns a CircuitInfo that other
methods can consume.

## Who reads a `CircuitInfo`, and file handling

- **Readers:** C-ΔΘ (`apply_ctheta`, `apply_ctheta_from_state_dicts`, `sweep_ctheta_strength`) reads
  `layer_suggestions.layer_subset` and `target_modules`; SafetyLoRA reads `layer_suggestions.target_modules`.
  LSSF, PKE, NLSR and DeepRefusal do not read it. No consumer reads the per-neuron `unit_ids`, so for C-ΔΘ the "circuit"
  is a set of layers and modules. `layer_subset` lists only the layers that kept units. `circuit_info=None` in C-ΔΘ
  means "every weight" (a Recover-pillar behaviour, unchanged).
- **Loading:** a missing file returns `None`; an unreadable or invalid file (bad JSON/YAML, an empty file, a list or scalar,
  an unknown key inside `safety_units` / `layer_suggestions`, `layer_indices` that are not non-negative integers) raises
  `ValueError` with the path and the reason. `{}` loads as an empty `CircuitInfo`. Unknown top-level keys of a CircuitKIT file are ignored.
- **Saving:** numpy / torch scalars are written as plain numbers, dict keys as strings; a NaN or infinite value raises
  `ValueError` and no file is written. `raw_output_path` is not saved.
- **`CircuitInfoProvider`** is a typing `Protocol` that nothing implements; `get_circuit_info` is a file loader that ignores
  `model_name` and every keyword except `top_fraction` and `layer_fraction`.
