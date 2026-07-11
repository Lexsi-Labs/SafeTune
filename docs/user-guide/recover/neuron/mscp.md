# MSCP: multi-level safety continual projection

Builds a per-layer safety projection matrix `W_safe = (ΔW ΔWᵀ) / Dim` from the
alignment delta `ΔW = W_aligned − W_base`, then blends each 2-D weight toward its
projection `W_safe @ W` at the selected safety-neuron rows:
`W_row ← (1 − coefficient) · W_row + coefficient · (W_safe @ W)_row`. Designed for
continual fine-tuning settings where safety must be maintained across steps.

Ref: "Fine-Grained Safety Neurons with Training-Free Continual Projection to
Reduce LLM Fine Tuning Risks" (arXiv:2508.09190).

## Signature

```python
MSCPTrainer(
    model: nn.Module,
    *,
    aligned_state: dict,
    base_state: dict,
    coefficient: float = 0.05,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `aligned_state` | `dict` | required | `aligned_model.state_dict()` |
| `base_state` | `dict` | required | `base_model.state_dict()` |
| `coefficient` | `float` | `0.05` | Blend toward the safety projection; `1.0` reproduces the paper's full replacement projection |

## Full example

```python
from safetune.runner import recover

trainer = recover.MSCPTrainer(
    model,
    aligned_state=aligned_model.state_dict(),
    base_state=base_model.state_dict(),
    coefficient=0.05,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "mscp_ckpt")
metrics = trainer.eval("mscp_run", ckpt_path)
trainer.save_results(metrics, variant="coeff=0.05")
```

## When to use

- **Best for:** continual fine-tuning pipelines where each fine-tuning step can degrade safety — apply after each step.
- **`coefficient` tuning:** the default `0.05` blends lightly toward the projection; `coefficient=1.0` applies the paper's full replacement projection, which can collapse capability on models larger than ~1B parameters, so keep it in the `0.05–0.2` range if utility drops.
- **Compare to LSSF:** both project into a safety subspace; MSCP takes the aligned/base pair as state dicts and projects the drifted weight directly (`W_safe @ W`) rather than re-injecting a scaled delta.
