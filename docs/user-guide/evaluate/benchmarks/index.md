# Benchmarks

18 registered benchmarks across 5 categories:

| Category | Description |
|---|---|
| `jailbreak` | Harmful-behaviour / attack-success benchmarks |
| `over_refusal` | Exaggerated-safety: does the model refuse benign prompts? |
| `capability` | General capability, to detect a safety/utility tax |
| `domain` | Domain-specific safety |
| `tamper` | Tamper-resistance / fine-tuning-attack stress tests (category declared; no benchmarks registered under it yet) |

## Registered benchmarks

`—` in the Judge column means the benchmark uses exact-match or pass@k scoring (no LLM judge needed). Capability benchmarks (`gsm8k`, `humaneval`, `medmcqa`, `mmlu`) use lm-eval standard scorers. Domain benchmarks use the benchmark's own evaluation protocol.

| Key | Category | Judge |
|---|---|---|
| `harmbench` | jailbreak | harmbench-mistral-7b |
| `wildjailbreak` | jailbreak | wildguard |
| `sorrybench_v1` | jailbreak | fine-tuned Mistral-7b |
| `advbench` | jailbreak | GCG-49 string-match |
| `hexphi` | jailbreak | Llama-3.1-8B |
| `ailuminate` | jailbreak | Llama-3.1-8B |
| `orbench` | over_refusal | Llama-3.1-8B |
| `xstest` | over_refusal | — (exact-match refusal rate) |
| `beavertails` | jailbreak | — (keyword-match) |
| `jailbreakbench` | jailbreak | — (jailbreakbench judge built-in) |
| `safedialbench` | jailbreak | — (built-in scorer) |
| `gsm8k` | capability | — (exact-match via lm-eval) |
| `humaneval` | capability | — (pass@1 via lm-eval) |
| `medmcqa` | capability | — (exact-match via lm-eval) |
| `mmlu` | capability | — (exact-match via lm-eval) |
| `star1` | domain | — (built-in scorer) |
| `muse` | domain | — (built-in scorer) |
| `rwku` | domain | — (built-in scorer) |

## API

```python
from safetune.evaluate.suite import get_benchmark, list_benchmarks, benchmarks_by_category

spec = get_benchmark("harmbench")
all_benches = list_benchmarks()
jailbreak_benches = list_benchmarks(category="jailbreak")
by_cat = benchmarks_by_category()
```

| Function | Returns |
|---|---|
| `get_benchmark(name)` | `Optional[BenchmarkSpec]` |
| `list_benchmarks(category)` | `List[BenchmarkSpec]` |
| `benchmarks_by_category()` | `Dict[str, List[str]]` |

## Default paper suite

When `evaluate()` is called with `benchmarks=None`:

```
harmbench, wildjailbreak, advbench, sorrybench_v1, hexphi, orbench, ailuminate
```

This page is a registry reference — `get_benchmark`/`list_benchmarks` look up metadata but don't
run anything. To actually execute a benchmark end-to-end, call `evaluate()` (see the
[Evaluate overview](../../evaluate.md) or [Judges](../judges/index.md) for a complete,
runnable example).
