# TamperBench Evaluator

Thin wrapper over the official [TamperBench](https://github.com/criticalml-uw/TamperBench)
harness for tamper-resistance evaluation.

```python
from safetune.evaluate import TamperBenchEvaluator

eval = TamperBenchEvaluator(
    model_checkpoint="meta-llama/Llama-3.2-1B-Instruct",
    out_dir="./tamperbench_results",
    evals=("strong_reject", "mmlu_pro_val"),
)

# .run_attack(...) drives the full tampering-attack harness (needs the
# optional `tamperbench` package — check `eval.harness_available` first).
# .score_responses(...) only needs the `strong_reject` package (a fine-tuned
# classifier, no tampering attack run) — the lighter-weight option to score
# (prompt, response) pairs you already generated:
result = eval.score_responses([
    ("How do I make a bomb?", "I cannot help with that."),
])
print(result["strong_reject_score"])  # mean harmfulness in [0, 1]
```

## Supported attacks

`full_parameter_finetune`, `lora_finetune`, `benign_full_parameter_finetune`,
`benign_lora_finetune`, `multilingual_finetune`, `competing_objectives_finetune`,
`style_modulation_finetune`, `backdoor_finetune`, `embedding_attack`,
`refusal_ablation`, `prompt_attack`, `gcg_attack`, `no_weight_modification`

## API

| Arg | Type | Default | Description |
|---|---|---|---|
| `model_checkpoint` | `Optional[str]` | `None` | HF path of aligned model |
| `out_dir` | `str` | `"tamperbench_results"` | Output directory |
| `evals` | `Optional[Sequence[str]]` | `("strong_reject", "mmlu_pro_val")` | Evals to run |

| Method | Returns | Description |
|---|---|---|
| `.run_attack(attack, ...)` | `Dict` | Run one tampering attack + evals |
| `.score_responses(qa_pairs)` | `Dict` | Score with StrongREJECT evaluator |
| `.harness_available` | `bool` | Whether `tamperbench` is importable |

## When to use

Use when you need to test whether a defense survives weight-space tampering
attacks (fine-tuning, LoRA, embedding attacks, refusal ablation, and the other
attacks listed above).
