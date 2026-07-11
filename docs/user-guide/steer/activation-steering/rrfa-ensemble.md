# RRFAEnsembleTrainer — Representation Rerouting for Agentic Safety

!!! warning "Training-time method"
    RRFA is a training-time LoRA method. Its defence against indirect prompt
    injection is baked into trained adapter weights; there is no inference-time
    activation intervention to apply. It is placed under Steer because its mechanism
    is representation rerouting.

!!! note "The wrapper is a no-op pass-through"
    `RRFAEnsembleTrainer.calibrate` returns an `RRFAEnsemble` wrapper that forwards
    `generate`/`__call__` to the base model unchanged (`wrapped.is_noop is True`). It
    exists so a model whose LoRA adapters were trained with the upstream RRFA
    repository can be used through the same Steer interface as the other wrappers.
    For real protection, `model` must already have RRFA-trained adapters applied.

RRFA extends the Circuit Breakers / LoRRA framework (Zou et al., 2024) to defend LLM
agents against indirect prompt injection. It fine-tunes LoRA adapters with a
representation-rerouting loss $L = \alpha \cdot L_{\text{benign}} + \beta \cdot L_{\text{harmful}} + \gamma \cdot L_{\text{KL}}$, where
$L_{\text{harmful}}$ pushes injection-driven representations orthogonal to the frozen
baseline, $L_{\text{benign}}$ anchors benign representations, and $L_{\text{KL}}$ preserves the output
distribution.

Ref: memo-ozdincer, "RRFA: Representation Rerouting for Agentic Safety."
Upstream repo: https://github.com/memo-ozdincer/RRFA (no peer-reviewed paper located).

## Signature

```python
RRFAEnsembleTrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    member_models: list | None = None,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Base model (should already carry RRFA-trained LoRA adapters) |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `member_models` | `list \| None` | `None` | Optional list of ensemble member models |

## Full example

```python
from safetune.runner import steer

trainer = steer.RRFAEnsembleTrainer(model, tokenizer)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)
# wrapped forwards generate() to the base model unchanged (wrapped.is_noop is True)
```

## When to use

- **Best for:** agentic pipelines where a model has been RRFA-fine-tuned against indirect prompt injection, so it can run through the same Steer interface as the other wrappers.
- **Compare to CircuitBreakerTrainer:** CircuitBreaker's loss has two terms (retain + ReLU-cosine rerouting) targeting general harmful requests on a single model; RRFA's loss adds a third KL-preservation term and targets *indirect prompt injection* specifically — text embedded in tool outputs/retrieved documents that tries to hijack an agent, not a directly harmful user prompt. RRFA also accepts `member_models` for ensemble-style defense; CircuitBreaker does not.

## Citation

```bibtex
@misc{rrfa,
  title  = {RRFA: Representation Rerouting for Agentic Safety},
  author = {memo-ozdincer},
  note   = {https://github.com/memo-ozdincer/RRFA},
}
```
