# Safe LoRA

Projects each fine-tuning weight update through a Safe-LoRA-style safety subspace
projection matrix (built from the aligned-minus-base delta). Per layer, it measures
how aligned the update is with that subspace and projects the layers that fall below
`threshold` toward the safety subspace, so the task update does not overwrite
alignment-critical weight directions. Applied at merge time — no training required.

Ref: Hsu et al., "Safe LoRA: The Silver Lining of Reducing Safety Risks when
Fine-tuning Large Language Models," NeurIPS 2024, arXiv:2405.16833.

## Signature

```python
SafeLoRATrainer(
    model: nn.Module,
    *,
    aligned_state_dict: dict,
    base_state_dict: dict,
    alpha: float = 0.5,
    threshold: float = 0.5,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Model with LoRA adapters already merged — modified in-place |
| `aligned_state_dict` | `dict` | required | State dict of the aligned model |
| `base_state_dict` | `dict` | required | State dict of the pre-alignment base model |
| `alpha` | `float` | `0.5` | Merge coefficient; also the fallback for `threshold` when `threshold` is not set |
| `threshold` | `float` | `0.5` | Alignment cutoff; layers whose update cosine falls below this are projected onto the safety subspace |

## Full example

```python
from safetune.runner import recover
from safetensors.torch import load_file

aligned_state_dict = load_file("./aligned/model.safetensors")
base_state_dict = load_file("./base/model.safetensors")

trainer = recover.SafeLoRATrainer(
    model,
    aligned_state_dict=aligned_state_dict,
    base_state_dict=base_state_dict,
    alpha=0.5,
    threshold=0.5,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "safelora_ckpt")
metrics = trainer.eval("safelora_run", ckpt_path)
trainer.save_results(metrics, variant="threshold=0.5")
```

## When to use

- **Best for:** post-LoRA-merge safety restoration when you know the drift came from a LoRA adapter.
- **Compare to LSSF:** both project onto a safety subspace; SafeLoRA gates per layer by how aligned each update is with that subspace.
- **Compare to RESTA:** RESTA adds the full delta; SafeLoRA projects the task update instead — less blunt, preserves capability directions.

## Citation

```bibtex
@inproceedings{hsu2024safelora,
  title     = {Safe LoRA: The Silver Lining of Reducing Safety Risks when Fine-tuning Large Language Models},
  author    = {Hsu, Chia-Yi and others},
  booktitle = {NeurIPS},
  year      = {2024},
  note      = {arXiv:2405.16833},
}
```
