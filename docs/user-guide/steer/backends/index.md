# Steering backends & unified runner

Generate from a steered model at scale. The `steer.run()` entry point dispatches to the correct backend.

| Backend | Speed | Use case |
|---|---|---|
| [Unified runner](runner.md) | — | Auto-dispatches based on wrapper type |
| [VLLMHookSteer](vllm-hook-steer.md) | ~6× HF | Activation-steering at scale |
| [VLLMDecodeSteer](vllm-decode-steer.md) | vLLM | Decoding-steering at scale |
| [SteerSpec](steer-spec.md) | — | Serializable steering spec |
| [DecodeSteerSpec](decode-steer-spec.md) | — | Serializable decoding spec |
