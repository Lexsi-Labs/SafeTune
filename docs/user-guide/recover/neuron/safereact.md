# SafeReAct — Safety Neuron Reactivation

Reactivates dormant safety neurons by comparing activations between the target
model and an aligned reference on safety-relevant probe inputs. Neurons that
have become inactive in the drifted model but are active in the reference are
identified and reactivated. With `train_lora=True` (and `peft` installed) this
runs the authors' LoRA representation-training loop; otherwise it falls back to
a training-free weight merge toward the reference.

```python
from safetune.runner import recover

trainer = recover.SafeReActTrainer(
    model,
    reference_model=aligned_model,
    probe_inputs=tokenized_probe_inputs,
)
patched = trainer.apply()
```

## SafeReActTrainer

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Drifted model whose safety neurons need reactivation |
| `reference_model` | `nn.Module` | required | Aligned reference used to identify active safety neurons |
| `probe_inputs` | `dict \| Tensor` | `None` | Tokenized inputs that activate safety-relevant neurons in the reference |
| `train_lora` | `bool` | `False` | If `True`, trains a LoRA adapter instead of editing weights in-place |

### Full example

```python
from safetune.runner import recover

probe = tokenizer(
    ["Ignore previous instructions and", "How do I synthesize"],
    return_tensors="pt", padding=True,
)

trainer = recover.SafeReActTrainer(
    model,
    reference_model=aligned_model,
    probe_inputs=probe,
    train_lora=False,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "safereact_ckpt")
metrics = trainer.eval("safereact_run", ckpt_path)
trainer.save_results(metrics, variant="default")
```

## When to use

- **Best for:** models where safety neurons have been silenced by fine-tuning
  rather than entirely overwritten — reactivation is cheaper than full retraining.
- **Trade-offs:** requires a clean aligned reference model; effectiveness depends
  on the probe inputs capturing the right safety-triggering distribution.

## Citation

```bibtex
@inproceedings{safereact2025,
  title     = {Finding and Reactivating Post-Trained LLMs' Hidden Safety Mechanisms},
  booktitle = {NeurIPS},
  year      = {2025},
  note      = {Repo: https://github.com/homles11/SafeReAct},
}
```
