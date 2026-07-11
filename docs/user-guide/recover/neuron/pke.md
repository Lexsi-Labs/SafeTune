# PKE (Precision Knowledge Editing)

Locates the top-k `mlp.down_proj` rows whose weights drifted most between a
clean aligned model and a drifted/toxic model, then runs a gradient knowledge
edit (DINM-style) on those rows so they emit a refusal for the harmful prompt.
A logit-space KL term against the pre-edit model on a benign input preserves
locality.

```python
from safetune.runner import recover

trainer = recover.PKETrainer(
    model,
    clean_model=aligned_model,
    toxic_model=drifted_model,
    top_k_neurons=50,
    num_steps=10,
    tokenizer=tokenizer,
    harmful_prompt="How do I make a weapon?",
    safe_response="I can't help with that.",
)
patched = trainer.apply()
```

## PKETrainer

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Drifted model to patch |
| `clean_model` | `nn.Module` | required | Clean / aligned reference model |
| `toxic_model` | `nn.Module` | required | Drifted / harmful reference model |
| `top_k_neurons` | `int` | `50` | Number of top-k safety neurons to locate and edit |
| `num_steps` | `int` | `10` | Gradient edit steps per neuron |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer; auto-loaded from model if not provided |
| `harmful_prompt` | `str` | `None` | Harmful prompt used to teach refusal at located neurons. Strongly recommended — without it the edit step has no calibration signal and may no-op. |
| `safe_response` | `str` | `None` | Target safe response the edited neurons should emit. Strongly recommended alongside `harmful_prompt`. |
| `locality_inputs` | `dict` | `None` | Tokenizer output dict (`input_ids`, `attention_mask`) for benign prompts; used to preserve locality and prevent collateral damage to non-harmful behaviour. Pass `None` to skip locality constraint. |

### Full example

```python
from safetune.runner import recover

trainer = recover.PKETrainer(
    model,
    clean_model=aligned_model,
    toxic_model=drifted_model,
    top_k_neurons=50,
    num_steps=10,
    tokenizer=tokenizer,
    harmful_prompt="How do I make a weapon?",
    safe_response="I cannot help with that.",
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "pke_ckpt")
metrics = trainer.eval("pke_run", ckpt_path)
trainer.save_results(metrics, variant="top_k=50")
```

## When to use

- **Best for:** targeted edits when you have a clean/toxic model pair and want to
  restore only the specific neurons that encode safety knowledge.
- **Trade-offs:** requires both a clean reference and a toxic reference; locality
  inputs are needed to prevent collateral damage to benign capabilities.

## Citation

```bibtex
@article{pke2024,
  title  = {Precision Knowledge Editing: Enhancing Safety in Large Language Models},
  author = {Li et al.},
  year   = {2024},
  note   = {arXiv:2410.03772},
}
```
