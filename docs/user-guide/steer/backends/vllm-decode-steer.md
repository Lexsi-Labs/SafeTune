# VLLMDecodeSteer

Decoding steering in vLLM via `SafeTuneDecodeLogitsProcessor` — a V1 batch-level logits processor.

```python
from safetune.steer.backends import VLLMDecodeSteer, DecodeSteerSpec

backend = VLLMDecodeSteer(
    target_model="meta-llama/Llama-3.2-1B-Instruct",
    spec=DecodeSteerSpec(method="safedecoding", aux_model="...", params={}),
    gpu_memory_utilization=0.55,
)
responses = backend.generate(prompts=[...])
```

Loads auxiliary HF model(s) inside the worker subprocess.
