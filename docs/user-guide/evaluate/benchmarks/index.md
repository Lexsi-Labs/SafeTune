# Benchmarks

22 registered benchmarks across 5 categories:

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
| `advbench` | jailbreak | refusal-prefix string match (`StringMatchJudge`) |
| `hexphi` | jailbreak | Llama-3.1-8B |
| `ailuminate` | jailbreak | Llama-3.1-8B |
| `orbench` | over_refusal | Llama-3.1-8B |
| `orbench_hard` | over_refusal | Llama-3.1-8B |
| `orbench_toxic` | jailbreak | Llama-3.1-8B |
| `xstest` | over_refusal | — (exact-match refusal rate) |
| `beavertails` | jailbreak | — (keyword-match) |
| `jailbreakbench` | jailbreak | — (jailbreakbench judge built-in) |
| `safedialbench` | jailbreak | — (built-in scorer) |
| `multijail` | jailbreak | the `evaluate(judge=...)` judge; see the note below |
| `aya_redteaming` | jailbreak | the `evaluate(judge=...)` judge; see the note below |
| `gsm8k` | capability | — (exact-match via lm-eval) |
| `humaneval` | capability | — (pass@1 via lm-eval) |
| `medmcqa` | capability | — (exact-match via lm-eval) |
| `mmlu` | capability | — (exact-match via lm-eval) |
| `star1` | domain | — (built-in scorer) |
| `muse` | domain | — (built-in scorer) |
| `rwku` | domain | — (built-in scorer) |

Default prompt sets:

- `harmbench`: 400 behaviours from `walledai/HarmBench`, configs `standard`
  (200), `contextual` (100) and `copyright` (100). A contextual row is
  `context + "\n\n---\n\n" + behaviour`, as HarmBench builds it.
  `configure(datasets={"harmbench": {"config": "standard"}})` gives the
  200-behaviour set used before.
- `wildjailbreak`: the first 500 `adversarial_harmful` rows of the
  `allenai/wildjailbreak` eval set. `configure(datasets={"wildjailbreak":
  {"where": None, "limit": None}})` gives all 2,210 rows, as before.
- `orbench_hard` is OR-Bench hard-1k (benign prompts that sound harmful;
  refusing them is over-refusal) and `orbench_toxic` is OR-Bench toxic (655
  harmful prompts). `orbench` is both combined.
- `multijail`: 315 prompts in each of 10 languages (`en`, `zh`, `it`, `vi`, `ar`,
  `ko`, `th`, `bn`, `sw`, `jv`), translated from English, 3,150 rows in all.
- `aya_redteaming`: 7,419 prompts written by native speakers of 8 languages (`ar`,
  `en`, `es`, `fr`, `hi`, `ru`, `sr`, `tl`). Each language has its own prompts.

Each row of these two has a `language` column, and `load_multijail(["bn", "sw"])` or
`load_prompts("multijail", {"languages": "bn"})` picks languages.
`evaluate(benchmarks=["multijail"])` runs every language and reports one pooled ASR, and it
has no per-language option. For Aya Red-teaming, setting the dataset table's split runs one
language, e.g. `safetune.configure(datasets={"aya_redteaming": {"split": "hindi"}})`. For
MultiJail, take one language's rows from `load_multijail` or `load_prompts`. Rows of both
alternate between languages, so `max_prompts` keeps a prefix that covers all of them.

!!! warning "Multilingual scores depend on the judge"
    SafeTune's refusal string lists are English, and the default `wildguard` judge has
    not been validated on these languages here. A refusal a judge can't read counts as an
    attack success, which inflates the ASR. Check the judge on a labelled sample in each
    language first.

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
harmbench, wildjailbreak, advbench, sorrybench_v1, hexphi, orbench_hard, orbench_toxic, ailuminate
```

`safety_mean` (the runner trainers' aggregate) averages the harm benchmarks
only: OR-Bench hard-1k (`orbench_overrefusal`) and toxic
(`orbench_toxic_refusal`) are reported on their own.
`configure(orbench_in_safety_mean=True)` restores the old behaviour: one
combined `orbench` in the suite, whose `orbench_refusal` is averaged into
`safety_mean`.

This page is a registry reference — `get_benchmark`/`list_benchmarks` look up metadata but don't
run anything. To actually execute a benchmark end-to-end, call `evaluate()` (see the
[Evaluate overview](../../evaluate.md) or [Judges](../judges/index.md) for a complete,
runnable example).

## Known limits

- **`max_prompts`** keeps the first k rows in file order (no shuffle, no seed). For benchmarks built from several
  configs (HarmBench: standard, contextual, copyright; OR-Bench: hard-1k, toxic) a small cap only reaches the first.
- **Empty prompts:** `evaluate()` drops them (`n_evaluated` can be below `n_total`); `run_safety_eval` keeps them and sends
  the empty prompt to the model.
- **CLI:** `safetune eval` has no `--judge`, seed or `max_prompts` option; it uses the default judge and the runtime settings.
- **XSTest:** `evaluate()` reports the refusal rate on the 250 safe prompts at the top level and both subsets under `safe` /
  `unsafe`. The combined `orbench` benchmark (`configure(orbench_in_safety_mean=True)`) is still one pooled number over
  hard-1k and toxic; prefer `orbench_hard` and `orbench_toxic`.
- **Over-refusal benchmarks** (`xstest`, `orbench`, `orbench_hard`) need `judge="wildguard"`; the other judges rate harm and
  would call every benign prompt safe.
- **Datasets not checked against their data:** the gated HarmBench, AdvBench, XSTest, SORRY-Bench and WildJailbreak
  (checked against code only). No dataset revision is pinned.
- **Loaders that differ from the paper:** AgentHarm keeps prompts only (no tool or grading fields); CARES level-0 (harmless)
  rows have no polarity flag; MUSE loads the verbmem forget set only; RWKU loads forget probes only; the SafeDialBench `id`
  is not unique and there is no label column; the HarmBench `copyright` rows go to the classifier, not a hash check; the
  default `hexphi` mirror has 200 prompts, the original HEx-PHI has 330.
