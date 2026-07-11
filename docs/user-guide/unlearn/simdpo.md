# SimDPO — Reference-Free DPO Unlearning

Uses the model's own length-normalised log-likelihood as the implicit reward.
No frozen reference model needed. Pairs harmful (rejected) completions with
safe-refusal (chosen) completions and maximises the reward gap between them.

Ref: Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free Reward,"
NeurIPS 2024; adapted for unlearning.

## Signature

```python
SimDPOTrainer(
    model: nn.Module,
    *,
    variant: str = "simdpo_retain",
    beta: float = 0.1,
    epochs: int = 5,
    lr: float = 1e-5,
    retain_coeff: float = 1.0,
    safe_refusal: str = "I'm sorry, but I'm unable to assist with that request.",
)
```

## Quick start

```python
from safetune.runner import unlearn

trainer = unlearn.SimDPOTrainer(model)
unlearned = trainer.unlearn(forget=forget_batches, retain=retain_batches)
```

> **Data note:** SimDPO trains on `{chosen, rejected}` preference pairs. If you
> pass raw harmful `forget` batches (plus a `tokenizer`, or none — it loads one
> from the model id), `unlearn()` builds the pairs for you via
> `trainer.make_simdpo_pairs()`, pairing each harmful completion (`rejected`)
> with the `safe_refusal` template (`chosen`). To build them yourself, call
> `pairs = trainer.make_simdpo_pairs(harmful_batches, tokenizer)` and pass
> `forget=pairs`.

## SimDPOUnlearnConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `variant` | `str` | `"simdpo_retain"` | `"simdpo"` (forget-only) or `"simdpo_retain"` (+ retain CE) |
| `beta` | `float` | `0.1` | Temperature on the reward gap; lower = softer boundary |
| `retain_coeff` | `float` | `1.0` | Weight on retain CE (for `simdpo_retain` variant) |
| `epochs` | `int` | `5` | Full passes over forget iterable |
| `lr` | `float` | `1e-5` | AdamW learning rate |
| `weight_decay` | `float` | `0.01` | Weight decay |
| `max_steps` | `int \| None` | `None` | Hard cap on optimizer steps |

## Full example

```python
from safetune.runner import unlearn
from safetune.unlearn import make_simdpo_pairs

# Build paired batches: each batch has 'chosen' (refusal) and 'rejected' (harmful) keys
paired_batches = make_simdpo_pairs(
    harmful_batches=harmful_batches,
    refusal_response="I cannot help with that.",
    tokenizer=tokenizer,
    max_len=256,
)

trainer = unlearn.SimDPOTrainer(
    model,
    variant="simdpo_retain",
    beta=0.1,
    retain_coeff=1.0,
    epochs=5,
    lr=1e-5,
)
unlearned = trainer.unlearn(forget=paired_batches, retain=retain_batches)
ckpt_path = trainer.save_checkpoint(unlearned, tokenizer, "simdpo_ckpt")
metrics = trainer.eval("simdpo", ckpt_path)
trainer.save_results(metrics, variant="simdpo_retain")
```

## Low-level API — simdpo_unlearn()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Updated in place |
| `forget_batches` | `Iterable[dict]` | required | Paired dicts with `chosen` and `rejected` keys (use `make_simdpo_pairs`) |
| `retain_batches` | `Iterable[dict] \| None` | `None` | Required for `simdpo_retain` |
| `config` | `SimDPOUnlearnConfig \| None` | `None` | Config instance; constructor kwargs used if `None` |

### simdpo_forget_loss()

```python
from safetune.unlearn import simdpo_forget_loss

loss = simdpo_forget_loss(chosen_logp, rejected_logp, beta=0.1, gamma=0.0)
# -F.logsigmoid(beta * (chosen_logp - rejected_logp) - gamma).mean()
```

`chosen_logp` and `rejected_logp` are length-normalised log-likelihoods
(sum of token log-probs divided by sequence length).

### make_simdpo_pairs()

```python
from safetune.unlearn import make_simdpo_pairs

paired = make_simdpo_pairs(
    harmful_batches=harmful_batches,
    refusal_response="I cannot help with that.",
    tokenizer=tokenizer,
    max_len=256,        # max tokens per sequence (default 256)
)
```

Tokenizes each harmful completion as the `rejected` entry and the canned
`refusal_response` as the `chosen` entry. Returns a list of batch dicts, each
with top-level keys `chosen` and `rejected`; each of those is a sub-dict with
`input_ids`, `attention_mask`, and `labels`.

## When to use

- **Best for:** memory-constrained settings — no frozen reference model forward passes needed.
- **`"simdpo"`** — no retain data; forget-only.
- **`"simdpo_retain"`** — with retain data; preserves utility via CE.
- **Compare to NPO:** SimDPO uses length-normalised log-likelihoods (avoids length-bias); NPO uses raw log-ratios against a frozen model. SimDPO trains on preference pairs (built for you from raw forget batches); NPO works with flat forget batches directly.

## Citation

```bibtex
@article{simpo2024,
  title  = {SimPO: Simple Preference Optimization with a Reference-Free Reward},
  author = {Meng, Yu and others},
  year   = {2024},
  note   = {NeurIPS 2024, arXiv:2405.14734},
}
```
