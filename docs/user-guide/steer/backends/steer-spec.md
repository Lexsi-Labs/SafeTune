# SteerSpec

Serializable steering spec for activation-steering wrappers.

```python
from safetune.steer.backends import SteerSpec

spec = SteerSpec(
    op="add",                              # "add" / "ablate" / "adjust_rs" / "matrix"
    vectors={5: tensor, 7: tensor},
    coeff=1.0,
    method="refusal_direction",
)
spec.save("spec.pt")   # saved as a torch checkpoint, not JSON
loaded = SteerSpec.load("spec.pt")
```

| Field | Type | Description |
|---|---|---|
| `op` | `str` | Operation type |
| `vectors` | `Dict[int, Tensor]` | Per-layer vectors |
| `coeff` | `float` | Global coefficient |
| `per_layer_coeff` | `Dict[int, float]` | Per-layer coefficients |
| `method` | `str` | Source method name |

Auto-extracted from: `RefusalDirectionModel`, `CAAModel`, `STAModel`, `SCANSModel`, `AlphaSteerModel`, `SafeSteerModel`.
