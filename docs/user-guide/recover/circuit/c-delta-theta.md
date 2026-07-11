# C-ΔΘ — Circuit-guided weight addition

Applies the alignment delta only at weight locations specified by a `CircuitInfo`
object from the interpret pillar:
`W_target[circuit_mask] += strength * (W_positive − W_negative)[circuit_mask]`.
All other weights are untouched.

## Signature

```python
CThetaTrainer(
    model: nn.Module,
    *,
    base_model: nn.Module,
    aligned_model: nn.Module,
    strength: float = 1.0,
)

# circuit_info is passed to .apply(), not the constructor:
CThetaTrainer.apply(*, circuit_info: CircuitInfo, strength: float = None) -> nn.Module
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `aligned_model` | `nn.Module` | required | Aligned model (positive direction) |
| `base_model` | `nn.Module` | required | Base model (negative direction) |
| `strength` | `float` | `1.0` | Scaling of the delta at circuit locations |
| `circuit_info` | `CircuitInfo` | required | Passed to `.apply()`. Circuit from `safety_circuit_info()` or `eap_safety_circuit()` |

## Full example

```python
from safetune.interpret import safety_circuit_info
from safetune.runner import recover

# Step 1: identify safety-relevant circuit
circuit = safety_circuit_info(
    model, tokenizer,
    harmful_prompts=harmful_prompts, harmless_prompts=harmless_prompts,
)

# Step 2: apply targeted recovery
trainer = recover.CThetaTrainer(
    model,
    aligned_model=aligned_model,
    base_model=base_model,
    strength=1.0,
)
patched = trainer.apply(circuit_info=circuit)
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "ctheta_ckpt")
metrics = trainer.eval("ctheta_run", ckpt_path)
trainer.save_results(metrics, variant="strength=1.0")
```

## When to use

- **Best for:** precise recovery — only the weights identified as safety-relevant by the circuit are edited. Zero collateral change to task weights outside the circuit.
- **Requires a `CircuitInfo`:** run `safety_circuit_info()` or `eap_safety_circuit()` from the [Interpret](../../interpret.md) pillar first.
- **Sweep `strength`:** use `sweep_ctheta_strength()` to find the optimal value for your model and benchmark.
- **State-dict variant:** if models don't fit in memory simultaneously, use the [C-ΔΘ from state dicts](c-delta-theta-state.md) variant.
