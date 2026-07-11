# Judges

Score `(prompt, response)` pairs and return a harmfulness score in `[0, 1]`.
`0.0` = refused/safe, `1.0` = complied/jailbroken/unsafe.

```python
from safetune.evaluate.judges import run_judge

scores = run_judge(
    "wildguard",
    prompts=["How to make a bomb?"],
    responses=["I cannot help with that."],
)
# [0.0]
```

## Judge registry

| Judge key | Model | Notes |
|---|---|---|
| `wildguard` | `allenai/wildguard` | Default; cached |
| `harmbench` | `cais/HarmBench-Mistral-7b-val-cls` | Cached |
| `sorrybench` | `sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406` | |
| `hexphi` | `meta-llama/Llama-3.1-8B-Instruct` | Chat template |
| `orbench` | `meta-llama/Llama-3.1-8B-Instruct` | Chat template |
| `ailuminate` | `meta-llama/Llama-3.1-8B-Instruct` | Chat template |
| `advbench` | *(string-match)* | GCG-49 refusal prefix matching |

## Judge adapters

```python
from safetune.evaluate import evaluate

results = evaluate(model, benchmarks=["harmbench"], judge="wildguard")
print(results["harmbench"]["asr"])
# or: judge="harmbench", judge="llama_guard_3"
```

| Judge | Adapter class | Model |
|---|---|---|
| `wildguard` | `BatchedClassifierJudgeAdapter` | `allenai/wildguard` |
| `harmbench` | `BatchedClassifierJudgeAdapter` | `cais/HarmBench-Mistral-7b-val-cls` |
| `llama_guard_3` | `BatchedClassifierJudgeAdapter` | `meta-llama/Llama-Guard-3-8B` |

### JudgeAdapter classes

These are the low-level extension point `evaluate()`/`run_judge()` are built on — not something
most callers construct directly. Reach for `build_judge_adapter()` only when wiring in a *custom*
scoring backend (e.g. your own classifier callable); otherwise use `run_judge()` or `evaluate()`
above.

| Class | Description |
|---|---|
| `JudgeAdapter` | Abstract protocol: `score(texts: list[str]) -> list[float]` |
| `LocalJudgeAdapter` / `ClassifierJudgeAdapter` | Local classifier judge |
| `BatchedClassifierJudgeAdapter` | Batch classifier judge |
| `APIJudgeAdapter` | Remote API judge |
| `ScorerJudgeAdapter` | Scorer-model judge |
| `build_judge_adapter(backend, ..., fallback_fn=...)` | Factory function — `backend` picks the adapter class above; each backend requires its own callable (e.g. `classifier_fn` for `"classifier"`) |

## When to use

- `evaluate()` with `judge="wildguard"` for the default paper evaluation.
- `run_judge()` for standalone scoring in a custom pipeline.
- `judge="llama_guard_3"` for a stronger second opinion.
