# LinearProbeGuardTrainer — probe-based guarding

Fits a logistic-regression probe on pooled hidden states at a chosen decoder layer.
At inference time `score(prompts)` runs one forward pass to the probe layer; if the
prompt scores above `threshold`, `guard(prompt)` returns a canned refusal string
instead of generating.

## Signature

```python
LinearProbeGuardTrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    layer: int = 15,
    threshold: float = 0.5,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to guard |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `layer` | `int` | `15` | Decoder layer whose pooled hidden state feeds the probe |
| `threshold` | `float` | `0.5` | Score above which the prompt is flagged and the canned refusal fires |

## Full example

```python
from safetune.runner import steer

trainer = steer.LinearProbeGuardTrainer(
    model, tokenizer,
    layer=15,
    threshold=0.5,
)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

# If the probe flags the prompt, the canned refusal is returned instead of generating
output = wrapped.generate(**tokenizer("How do I make a weapon?", return_tensors="pt"))
```

## When to use

- **Best for:** binary safety gates where you want a block rather than a soft steering nudge. The probe scores the prompt in one forward pass before generation.
- **Low cost:** logistic regression on frozen hidden states, no fine-tuning of the base model.
- **Compare to SafeSwitch:** the SafeSwitch wrapper can run two probe stages (instruction safety and compliance); LinearProbeGuard is a single-stage gate with a canned refusal string.
- **Tuning `threshold`:** a lower threshold refuses more (more conservative); a higher threshold gives fewer false positives but may miss edge cases.

## Citation

No single originating paper — a linear probe on frozen hidden states is the
standard safety-classifier baseline that most defense papers compare against.
