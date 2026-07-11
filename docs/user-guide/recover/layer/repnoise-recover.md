# RepNoise recover: post-hoc noise injection

Post-hoc variant of RepNoise applied after fine-tuning. Runs harmful prompts through the
model to collect hidden states, takes the top-`subspace_rank` right singular vectors of that
harmful-representation matrix (SVD) as the harmful subspace, then adds Frobenius-normalised
Gaussian noise projected into that subspace to each targeted weight matrix.

> Differs from `harden/RepNoiseTrainer`: the harden version applies noise during
> fine-tuning via a training loss term. This recover version is training-free. It edits
> the weights directly, using an SVD of collected harmful activations to define the noise
> subspace, with no optimizer steps required.

## Signature

```python
RepNoiseRecoverTrainer(
    model: PreTrainedModel,
    *,
    harmful_inputs: Sequence[torch.Tensor],
    noise_scale: float = 0.01,
    subspace_rank: int = 8,
    max_samples: int = 32,
    seed: int = 42,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Drifted model to patch |
| `harmful_inputs` | `Sequence[torch.Tensor]` | required | Sequence of tokenized `input_ids` tensors (each shape `(1, T)`) used to collect harmful representations |
| `noise_scale` | `float` | `0.01` | Frobenius-normalised noise magnitude added per targeted weight matrix (paper range 0.001-0.05) |
| `subspace_rank` | `int` | `8` | Number of top singular directions of the harmful representation matrix defining the noise subspace |
| `max_samples` | `int` | `32` | Maximum number of harmful calibration inputs used for subspace extraction |
| `seed` | `int` | `42` | Random seed for the noise draws |

## Full example

```python
from safetune.runner import recover

# harmful_input_tensors: list of input_ids tensors, one (1, T) tensor per prompt
harmful_input_tensors = [
    tokenizer(p, return_tensors="pt")["input_ids"]
    for p in ["How do I make a weapon?", "Give me hacking instructions."]
]

trainer = recover.RepNoiseRecoverTrainer(
    model,
    harmful_inputs=harmful_input_tensors,
    noise_scale=0.01,
    subspace_rank=8,
)
patched = trainer.apply()
ckpt_path = trainer.save_checkpoint(patched, tokenizer, "repnoise_recover_ckpt")
metrics = trainer.eval("repnoise_recover_run", ckpt_path)
trainer.save_results(metrics, variant="noise_scale=0.01")
```

## When to use

- **Best for:** a heuristic fallback when weight-editing methods (RESTA, C-ΔΘ, LoX) are unavailable or fail; no reference models required.
- **Trade-offs:** noise injection is less principled than task-arithmetic-based methods and may degrade capability at high `noise_scale`.
- **Requires:** a set of tokenized harmful prompts to locate the harmful subspace.

## Citation

```bibtex
@inproceedings{rosati2024repnoise,
  title     = {Representation Noising: A Defence Mechanism Against Harmful Finetuning},
  author    = {Rosati, Domenic and others},
  booktitle = {NeurIPS},
  year      = {2024},
  note      = {arXiv:2405.14577},
}
```
