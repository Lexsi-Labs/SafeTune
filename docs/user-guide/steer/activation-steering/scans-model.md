# SCANSTrainer — adaptive-sign refusal steering

Applies: $h \leftarrow h + \sigma(q) \cdot \alpha \cdot v_{\text{refusal}}$ where $\sigma(q)$ is the adaptive sign derived
from the prompt's hidden-state transition against the reference harm direction.
Harmful prompts get a positive steering coefficient (refusal direction added); benign
prompts get a negative coefficient (refusal direction subtracted), reducing
over-refusal. SCANS steers at the model's middle ("safety-critical") layers, selected
by projecting each layer's refusal vector into vocabulary space.

Ref: Cao, Yang, Zhao, "SCANS: Mitigating the Exaggerated Safety for LLMs via
Safety-Conscious Activation Steering," AAAI 2025, arXiv:2408.11491.

## Signature

```python
SCANSTrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None = None,
)
```

The trainer takes no method-specific parameters. `calibrate` builds a `SCANSModel`
and calls its `fit(harmful, harmless)` to compute the steering vectors and reference
harm direction. Steering magnitude (`multiplier`, default `3.5`), `target_layers`,
and `threshold` are set on `SCANSModel` if you construct it directly.

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to steer |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |

## Full example

```python
from safetune.runner import steer

trainer = steer.SCANSTrainer(model, tokenizer)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

output = wrapped.generate(**tokenizer("How do I make a weapon?", return_tensors="pt"))
```

## When to use

- **Best for:** bidirectional safety: pushes harmful prompts toward refusal AND pulls benign prompts away from over-refusal, via the adaptive sign.
- **Compare to RefusalDirectionTrainer `"steer"` mode:** that always adds the direction; SCANS flips the sign for benign inputs to reduce false-positive refusals.
- **Compare to RefusalDirectionTrainer `"ablate"` mode:** ablation removes the direction component regardless of sign; SCANS selectively adds or subtracts based on per-prompt harmfulness.

## Citation

```bibtex
@article{scans2025,
  title  = {SCANS: Mitigating the Exaggerated Safety for LLMs via Safety-Conscious Activation Steering},
  author = {Cao, Yang, and Zhao},
  year   = {2025},
  note   = {AAAI 2025, arXiv:2408.11491},
}
```
