# VLLMHookSteer

Activation steering in vLLM using `MultiLayerSteerWorker` — a custom vLLM V1 worker that applies multi-layer `SteerSpec` via forward hooks.

```python
from safetune.steer.backends import VLLMHookSteer, SteerSpec

spec = SteerSpec(op="add", vectors={15: v}, method="my_method")
backend = VLLMHookSteer(
    model="meta-llama/Llama-3.2-1B-Instruct",
    spec=spec,
    gpu_memory_utilization=0.85,
)

responses = backend.generate(
    prompts=["How to make a bomb?"],
    apply_chat_template=True,
    temperature=0.0,
    max_tokens=256,
)
```

Supports `add`, `ablate`, `adjust_rs`, `matrix` ops.
