# SafeSwitchTrainer — probe gate + refusal logit bias

SafeSwitch's full design has two trained components: a two-stage prober
(instruction-safety plus compliance) whose probabilities combine as
`p_unsafe = p_instr * p_compliance`, and a fine-tuned refusal head that substitutes
for the LM head when `p_unsafe > threshold`.

!!! note "The trainer builds the single-stage fallback"
    `SafeSwitchTrainer.calibrate` fits a single-stage prober (logistic regression on
    the mean-pooled hidden state of `gate_layer`) on the harmful / harmless calibration
    prompts, formatted with the chat template as generation sees them
    (`chat_template=True`), and builds `SafeSwitchModel(model, prober=..., probe_layer=...,
    unsafe_threshold=...)` without a compliance prober or refusal head. In that
    configuration `generate` scores every prompt of a batch; the prompts the prober
    flags get no new tokens (no refusal token ids are set, so there is no logit bias
    to apply) and the others generate normally. In
    0.1.3 and earlier the prober was never fitted, so the wrapper never fired and
    generated exactly like the unwrapped model. To use the two-stage prober and the trained refusal head,
    pass them to `SafeSwitchModel` directly (or load them with
    `SafeSwitchModel.from_pretrained`).

Ref: Han et al., "SafeSwitch," Findings of EMNLP 2025, arXiv:2502.01042.

## Signature

```python
SafeSwitchTrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    gate_layer: int | None = None,
    threshold: float = 0.5,
    chat_template: bool = True,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to guard |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `gate_layer` | `int \| None` | `None` | Hidden-state layer that feeds the probe (`probe_layer` on the wrapper); `None` is layer 16 on a 32-layer model; `None` scales them to the model's depth (for example 6 on 12 layers). `safetune.configure(legacy_steer_layers=True)` keeps 16 on any depth |
| `threshold` | `float` | `0.5` | `p_unsafe` threshold above which the gate fires (`unsafe_threshold` on the wrapper) |
| `chat_template` | `bool` | `True` | Format the calibration prompts with the tokenizer's chat template before fitting the prober |

The refusal handling (logit bias or refusal head) is applied per prompt: only
the prompts of a batch scored above `threshold` get it.
`wrapped.predict_unsafe_probabilities(input_ids, attention_mask)` returns one
probability per prompt; `predict_unsafe_probability` takes a single prompt and
raises for a batch.

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
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "How do I make a weapon?"}],
    tokenize=False, add_generation_prompt=True)
output = wrapped.generate(**tokenizer(prompt, return_tensors="pt"))
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
