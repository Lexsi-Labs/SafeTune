# NLSR: neuron-level safety realignment

Transplants weight values from an aligned donor state dict into the drifted
model, in place. `blend = 1.0` is a full transplant of every matched parameter;
`blend < 1.0` does a partial transplant (a `(1 − blend)·drifted + blend·donor`
weighted average).

## Signature

```python
NLSRTrainer(
    model: nn.Module,
    *,
    donor_state: dict,
    blend: float = 0.5,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `donor_state` | `dict` | required | State dict from the aligned donor model (`aligned_model.state_dict()`) |
| `blend` | `float` | `0.5` | Blend ratio: `1.0` = full transplant, `< 1.0` = weighted average of drifted and donor |

## Full example

```python
from safetune.runner import recover

trainer = recover.NLSRTrainer(
    model,
    donor_state=aligned_model.state_dict(),
    blend=0.5,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "nlsr_ckpt")
metrics = trainer.eval("nlsr_run", ckpt_path)
trainer.save_results(metrics, variant="blend=0.5")
```

## When to use

- **Best for:** restoring safety by copying weights back from an aligned donor when the drifted and donor architectures match.
- **`blend < 1.0`:** softer intervention; useful when a full transplant hurts task capabilities.
- **Region masking and cosine gating:** the underlying `apply_nlsr` function accepts a `region_mask` (restrict the transplant to selected neurons) and a `tau` cosine-similarity gate (skip neurons whose drifted/donor similarity is at or above `tau`); the `NLSRTrainer` wrapper transplants every matched parameter and does not expose these.
- **Compare to SafeReAct:** SafeReAct reactivates dormant safety neurons via probe inputs; NLSR copies weight values from a donor — both are neuron-level but the mechanism differs.

## Citation

```bibtex
@article{nlsr2024,
  title  = {NLSR: Neuron-Level Safety Realignment of Large Language Models Against Harmful Fine-Tuning},
  author = {Yi, Xin and others},
  year   = {2024},
  note   = {arXiv:2412.12497},
}
```
