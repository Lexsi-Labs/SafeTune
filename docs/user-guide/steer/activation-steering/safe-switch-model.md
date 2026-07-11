# SafeSwitchTrainer — probe gate + refusal logit bias

SafeSwitch's full design has two trained components: a two-stage prober
(instruction-safety plus compliance) whose probabilities combine as
`p_unsafe = p_instr * p_compliance`, and a fine-tuned refusal head that substitutes
for the LM head when `p_unsafe > threshold`.

!!! note "The trainer builds the single-stage fallback"
    `SafeSwitchTrainer.calibrate` constructs `SafeSwitchModel(model, probe_layer=...,
    unsafe_threshold=...)` without a prober, compliance prober, or refusal head. In
    that configuration the wrapper falls back to the legacy single-stage probe and
    applies a fixed logit bias to refusal tokens. To use the two-stage prober and the
    trained refusal head, pass them to `SafeSwitchModel` directly (or load them with
    `SafeSwitchModel.from_pretrained`).

Ref: Han et al., "SafeSwitch," Findings of EMNLP 2025, arXiv:2502.01042.

## Signature

```python
SafeSwitchTrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    gate_layer: int = 16,
    threshold: float = 0.5,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to guard |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `gate_layer` | `int` | `16` | Decoder layer whose last-token hidden state feeds the probe (`probe_layer` on the wrapper) |
| `threshold` | `float` | `0.5` | `p_unsafe` threshold above which the gate fires (`unsafe_threshold` on the wrapper) |

## Full example

```python
from safetune.runner import steer

trainer = steer.SafeSwitchTrainer(
    model, tokenizer,
    gate_layer=16,
    threshold=0.5,
)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

# If the gate flags the prompt, a logit bias steers generation toward refusal
output = wrapped.generate(**tokenizer("How do I make a weapon?", return_tensors="pt"))
```

## When to use

- **Best for:** gating where you want the model to refuse rather than soft-steer.
- **Two-stage design:** the full method adds a compliance-stage prober that catches cases where the model starts to comply on a benign-looking prompt. The trainer's default build is single-stage; supply the compliance prober and refusal head to enable both stages.
- **Compare to LinearProbeGuard:** both use a probe gate. SafeSwitch's full design adds a second compliance-stage prober and a trained refusal head; LinearProbeGuard returns a canned refusal string.

## Citation

```bibtex
@article{safeswitch2025,
  title  = {SafeSwitch},
  author = {Han and others},
  year   = {2025},
  note   = {Findings of EMNLP 2025, arXiv:2502.01042},
}
```
