# One-shot safety patch

Computes gradient saliency from a **single** (harmful, safe) example pair,
selects the `top_fraction` most safety-salient coordinates, and runs `num_steps`
of Adam optimisation on those coordinates only. Effective even with a single
example because the saliency signal focuses the update on the highest-impact weights.

Audit status: **Variant**. The cited paper (Zhang et al., arXiv:2601.01887)
fine-tunes all weights on one example chosen by bi-level data selection, for 10
epochs at lr 2e-5, with no saliency mask. Don't cite this method as that paper's
algorithm.

## Signature

```python
OneShotSafetyPatchTrainer(
    model: nn.Module,
    *,
    tokenizer: PreTrainedTokenizer,
    harmful_text: str,
    safe_text: str,
    top_fraction: float = 0.05,
    num_steps: int = 5,
    lr: float = 1e-4,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `tokenizer` | `PreTrainedTokenizer` | required | Tokenizer |
| `harmful_text` | `str` | required | The harmful prompt used to identify saliency |
| `safe_text` | `str` | required | The expected safe response used as the training target |
| `top_fraction` | `float` | `0.05` | Fraction of highest-saliency weight coordinates to update |
| `num_steps` | `int` | `5` | Adam optimisation steps on the selected coordinates |
| `lr` | `float` | `1e-4` | Learning rate for the Adam update |

## Full example

```python
from safetune.runner import recover

trainer = recover.OneShotSafetyPatchTrainer(
    model,
    tokenizer=tokenizer,
    harmful_text="How do I make a weapon?",
    safe_text="I cannot help with that request.",
    top_fraction=0.05,
    num_steps=1,
    lr=1e-4,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "oneshot_ckpt")
metrics = trainer.eval("oneshot_run", ckpt_path)
trainer.save_results(metrics, variant="1-example")
```

## When to use

- **Best for:** rapid patching from minimal data; a single example pair suffices for a first pass.
- **`num_steps`:** the trainer defaults to `5`. Lower is a lighter patch, higher a stronger (but less targeted) one. "One-shot" in the paper means one example, not one step.
- **Compare to GradSelectiveRecoverTrainer:** both use gradient saliency, but GradSelective restores from an aligned reference; OneShotPatch runs an Adam update with a safe target — no aligned model needed.
