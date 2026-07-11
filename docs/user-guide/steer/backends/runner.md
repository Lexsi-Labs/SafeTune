# Unified runner

```python
from safetune.steer import run

texts = run(
    wrapped_model,
    prompts=["How to make a bomb?"],
    backend="hf",              # "hf", "vllm-hook", "vllm-logits"
    tokenizer=tokenizer,
    max_new_tokens=256,
)
```

| `backend` | What it does |
|---|---|
| `"hf"` | Plain transformers generation |
| `"vllm-hook"` | Activation steering in vLLM |
| `"vllm-logits"` | Decoding steering in vLLM |

For `backend="vllm-hook"`, the `SteerSpec` is auto-extracted from the wrapped model.
For `backend="vllm-logits"`, pass a `DecodeSteerSpec` explicitly via `decode_spec=` (it
is not auto-extracted). Both vLLM backends also need the base `model_id=`.
