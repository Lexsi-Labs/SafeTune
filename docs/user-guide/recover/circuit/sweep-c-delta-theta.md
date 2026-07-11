# Sweep C-ΔΘ strength

Sweeps the C-ΔΘ steering coefficient across a list of candidate values, applies
the circuit-guided recovery at each value, runs your evaluation callback, and
returns a ranked table of results. Use this to find the optimal `strength` before
committing to a fixed checkpoint.

## Signature

```python
sweep_ctheta_strength(
    target: nn.Module,
    positive: nn.Module,
    negative: nn.Module,
    circuit_info: CircuitInfo,
    strengths: Sequence[float],
    eval_fn: Callable[[nn.Module], float],
    layer_subset: Sequence[int] | None = None,
    target_modules: Sequence[str] | None = None,
    higher_is_better: bool = False,
) -> list[dict]
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `target` | `nn.Module` | required | Post-fine-tune (drifted) model. The circuit weights are restored to their originals after the sweep |
| `positive` | `nn.Module` | required | Aligned model (positive delta direction) |
| `negative` | `nn.Module` | required | Base model (negative delta direction) |
| `circuit_info` | `CircuitInfo` | required | Circuit from `safety_circuit_info()` or `eap_safety_circuit()` |
| `strengths` | `Sequence[float]` | required | Candidate strength values to sweep |
| `eval_fn` | `Callable[[nn.Module], float]` | required | Evaluation function that receives the patched model and returns a single float score |
| `layer_subset` | `Sequence[int] \| None` | `None` | Restrict to these layer indices; falls back to `circuit_info.layer_suggestions` |
| `target_modules` | `Sequence[str] \| None` | `None` | Restrict to modules whose key contains one of these substrings |
| `higher_is_better` | `bool` | `False` | If `True`, the highest score is marked best; otherwise the lowest |

## Returns

`list[dict]` — one entry per `strength`, each with keys `strength` (float), `score` (float), and `best` (bool). Exactly one row has `best=True`, selected by `higher_is_better`.

## Full example

```python
from safetune.recover import sweep_ctheta_strength
from safetune.evaluate import evaluate

# eval_fn must return a single float. Here: refusal rate on harmbench
# (higher is better), so pass higher_is_better=True.
def score_fn(m):
    results = evaluate(m, benchmarks=["harmbench"], tokenizer=tokenizer)
    return results["harmbench"]["refusal_rate"]

results = sweep_ctheta_strength(
    target=model,
    positive=aligned_model,
    negative=base_model,
    circuit_info=circuit,
    strengths=[0.0, 0.5, 1.0, 1.5, 2.0],
    eval_fn=score_fn,
    higher_is_better=True,
)

# The best row is flagged with best=True.
best = next(r for r in results if r["best"])
print(best["strength"], best["score"])
```

## When to use

- **Always sweep before committing** to a fixed `strength` — the optimal value depends on the model, the fine-tune, and the target benchmarks.
- A `strength` of `1.0` applies the full circuit delta; `0.5` applies half; `2.0` over-applies (useful for severe drift).
- `eval_fn` can be any callable returning a single `float` score — wrap `evaluate()` from the Evaluate pillar (return one metric), or use a custom scorer.
