# Configuration

`SafeTuneConfig` is the shared configuration object behind the CLI's `--config`
flag and the YAML workflow. Load it from YAML or build it in Python; every field
maps to a CLI flag, and explicit CLI flags override config values.

```python
from safetune.config import SafeTuneConfig

cfg = SafeTuneConfig.from_yaml("run.yaml")
```

See the [CLI Reference](../cli.md) for the YAML schema and precedence rules.

## Runtime settings: `safetune.configure()`

`safetune.configure(**settings)` sets process-wide runtime settings
(`RuntimeConfig`), and `safetune.get_config()` returns the effective values. A
function or trainer argument passed explicitly wins over `configure()` (or the
`runtime:` block of a YAML config), which wins over `SAFETUNE_*` environment
variables and the defaults. An unknown setting name raises `TypeError`.

Defaults to be aware of:

- Evaluation is strict (`eval_strict=True`): a benchmark that fails to load or
  score raises. `configure(eval_strict=False)` records
  `{"error": "<type>: <message>"}` for that benchmark and continues.
- A misspelled or unknown keyword argument to a runner trainer produces a
  warning (not an error) and is ignored.
- `dtype="auto"` is bf16 on Apple Silicon (MPS) with macOS 14 or newer, fp32 on
  older macOS, and fp32 on CPU. On CUDA it is bf16 where the GPU supports it
  natively, else fp16.

### `legacy_*` settings

Each one restores the behaviour from before a fix, to reproduce earlier
results. All default to `False`.

| Setting | `True` gives |
|---|---|
| `legacy_beavertails_splits` | TAR adversary set `unsafe[n:2n]`, which overlaps the contamination set |
| `legacy_steer_layers` | Steer layer defaults as absolute indices (14–18, 15, 16, 10–19) on any depth |
| `legacy_cast_gate` | CAST gate fitted on raw prompts; the first prompt of a batch gates all of them |
| `legacy_spectral_monitor` | `SpectralEntropyMonitor` on raw prompts, with the first (attention-sink) token in the SVD |
| `legacy_ga_forget_clip` | `GradientAscentTrainer` / `GradDiffTrainer` default `forget_clip=0.5` |
| `legacy_constrained_sft` | Runner / CLI `ConstrainedSFTTrainer` trains without its reference model (plain SFT) |
| `legacy_alphasteer_layers` | AlphaSteer hooks the matrix fitted on `layers[i]` at decoder layer `i` |
| `legacy_derta` | DeRTa MLE over the harmful prefix too, plus RTO on prefix + refusal rows as a second loss |
| `legacy_resta_drop_rate` | ReSta DARE drop rate 0.9 instead of the paper's 0.3 |

Other settings that restore earlier defaults: `advbench_scorer="gcg"` (the
29-phrase GCG string match), `orbench_in_safety_mean=True` (one combined
OR-Bench averaged into `safety_mean`), and `datasets=` overrides for the
HarmBench and WildJailbreak prompt sets (see
[Benchmarks](../../user-guide/evaluate/benchmarks/index.md)).

## Reference

::: safetune.config.SafeTuneConfig
    options:
      show_source: false
      heading_level: 3
      members_order: source
