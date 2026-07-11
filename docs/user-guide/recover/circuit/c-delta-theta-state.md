# C-ΔΘ from state dicts

State-dict variant of C-ΔΘ that accepts pre-loaded state dicts instead of model
objects. Useful in distributed settings or when the positive and negative
reference models do not fit in GPU memory alongside the target model. This
variant is the function `apply_ctheta_from_state_dicts`, not a trainer class.

## Signature

```python
apply_ctheta_from_state_dicts(
    target: nn.Module,
    positive_sd: dict,
    negative_sd: dict,
    circuit_info: CircuitInfo,
    strength: float = 1.0,
    layer_subset: list[int] | None = None,
    target_modules: list[str] | None = None,
) -> nn.Module
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `target` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `positive_sd` | `dict` | required | Aligned model state dict (`aligned_model.state_dict()`) |
| `negative_sd` | `dict` | required | Base model state dict (`base_model.state_dict()`) |
| `circuit_info` | `CircuitInfo` | required | Circuit from `safety_circuit_info()` or `eap_safety_circuit()` |
| `strength` | `float` | `1.0` | Scaling of the delta at circuit locations |
| `layer_subset` | `list[int] \| None` | `None` | Restrict to these layer indices; falls back to `circuit_info.layer_suggestions` |
| `target_modules` | `list[str] \| None` | `None` | Restrict to modules whose key contains one of these substrings |

## Full example

```python
import torch
from safetune.interpret import safety_circuit_info
from safetune.recover import apply_ctheta_from_state_dicts

# Load state dicts (CPU-side to save GPU memory)
aligned_sd = torch.load("./aligned/model.bin", map_location="cpu")
base_sd    = torch.load("./base/model.bin",    map_location="cpu")

circuit = safety_circuit_info(
    model, tokenizer,
    harmful_prompts=harmful_prompts, harmless_prompts=harmless_prompts,
)

patched = apply_ctheta_from_state_dicts(
    model,
    positive_sd=aligned_sd,
    negative_sd=base_sd,
    circuit_info=circuit,
    strength=1.0,
)
```

## When to use

- **Memory-constrained settings:** load state dicts on CPU while the drifted model runs on GPU.
- **Distributed pipelines:** state dicts can be sent over the network or loaded from separate checkpoints.
- Produces the same edit as the model-object variant — use whichever is more convenient for your setup.
- See [C-ΔΘ (model objects)](c-delta-theta.md) for the trainer-based version.
