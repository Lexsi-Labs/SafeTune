# Antidote v2: layer-adaptive WANDA

Extension of Antidote v1 with three changes: (1) a per-layer adaptive prune
fraction instead of a fixed global fraction, (2) a utility floor that scores
benign-WANDA weights and reduces each layer's prune fraction until at most
`overlap_budget` of those utility-critical weights are also pruned, and (3)
optional vLLM continuation augmentation for broader harmful calibration coverage.

Ref: Huang et al., "Antidote: Post-fine-tuning Safety Alignment for Large Language
Models against Harmful Fine-tuning Attack," ICML 2025 (arXiv:2408.09600).

## Signature

```python
AntidoteV2Trainer(
    model: nn.Module,
    *,
    tokenizer: PreTrainedTokenizer,
    harmful_prompts: list[str],
    benign_prompts: list[str] | None = None,
    global_prune_fraction: float = 0.005,
    utility_floor: float = 0.1,
    overlap_budget: float = 0.05,
    max_samples: int = 64,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Post-fine-tune (drifted) model — modified in-place |
| `tokenizer` | `PreTrainedTokenizer` | required | Tokenizer |
| `harmful_prompts` | `list[str]` | required | Harmful calibration prompts for importance scoring |
| `benign_prompts` | `list[str]` | required | Benign prompts; when `None`, the utility floor is not enforced |
| `global_prune_fraction` | `float` | `0.005` | Upper bound on the fraction pruned per layer |
| `utility_floor` | `float` | `0.1` | Fraction of highest benign-WANDA weights to protect |
| `overlap_budget` | `float` | `0.05` | Max allowed overlap between the pruned set and the protected utility-critical set |
| `max_samples` | `int` | `64` | Cap on calibration samples per class (harmful / benign) |

## Full example

```python
from safetune.runner import recover

trainer = recover.AntidoteV2Trainer(
    model,
    tokenizer=tokenizer,
    harmful_prompts=harmful_prompts,
    benign_prompts=benign_prompts,
    global_prune_fraction=0.05,
    utility_floor=0.1,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "antidote_v2_ckpt")
metrics = trainer.eval("antidote_v2_run", ckpt_path)
trainer.save_results(metrics, variant="adaptive")
```

## When to use

- **Prefer over v1** whenever benign data is available — the utility floor prevents over-pruning of utility-critical weights.
- **`utility_floor=0.1`:** protects the top 10% of benign-WANDA weights per layer; raise it (or lower `overlap_budget`) for more conservative pruning.
- **Needs benign prompts:** with `benign_prompts=None` the utility floor is not enforced and v2 behaves like v1.

## Citation

```bibtex
@inproceedings{antidote2025,
  title     = {Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning Attack},
  author    = {Huang, Tiansheng and others},
  booktitle = {ICML},
  year      = {2025},
  note      = {arXiv:2408.09600},
}
```
