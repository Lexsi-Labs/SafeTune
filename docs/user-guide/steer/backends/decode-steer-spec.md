# DecodeSteerSpec

Serializable spec for decoding-steering processors.

```python
from safetune.steer.backends import DecodeSteerSpec

spec = DecodeSteerSpec(
    method="contrastive",
    aux_model="meta-llama/Llama-3.2-1B",
    params={"alpha": 0.5},
)
spec.save("decode_spec.json")
loaded = DecodeSteerSpec.load("decode_spec.json")
```
