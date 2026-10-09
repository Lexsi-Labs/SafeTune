# Changelog

All notable changes to SafeTune are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- `trl` is capped below 1.15. TRL 1.15.0 (8 Oct 2026) computes the DPO loss with a
  Triton kernel and does not check that the tensors are on a GPU, so on a Linux
  machine without one the DPO-based trainers (CST, DeRTa and the rest of that
  family) failed with `RuntimeError: 0 active drivers ([]). There should only be one.`
  The requirement is now `trl>=0.12,<1.15`.

## [0.1.8] - 2026-10-02

0.1.8, the version the release pipeline publishes next; the last PyPI
release was 0.1.7.

- The logo renders on the PyPI project page again. `README.md` pointed at it
  with a repo-relative path, which GitHub resolves inside a README but PyPI
  cannot — there is no repo checkout for a relative path to resolve against —
  so the image 404s on the package page. It now points at
  `raw.githubusercontent.com`, the same pattern AgentTune uses. The `LICENSE.md`
  href had the same root cause and is fixed with it.
- The declared version is reconciled with the index. `main` still said 0.1.6
  while PyPI served 0.1.7, because the 0.1.7 bump the pipeline made at release
  time never landed back on `main`. `scripts/check_version_consistency.py
  --pypi` had been failing on this the whole time; CI runs it without
  `--pypi`, so the drift was invisible.

0.1.7 on the index is a re-publish of 0.1.6 — the two sdists are identical
apart from their `dist-info` directory — so it gets no section of its own.

## [0.1.6] - 2026-10-01

0.1.6, the version the release pipeline publishes next; the last PyPI
release was 0.1.5. The `[Legacy numbering]` section further down holds the
pre-June-2026 numbering that peaked at 0.6.0 — none of it is on the index.

### Pull requests in this release

SafeTune-Internal pull requests, in stack order.

- ST-05 (#3) One settings mechanism for SafeTune (`configure()` + YAML),
  hardware and silent-ignore fixes.
- ST-06 (#4) One class per method, keyword-safe merge functions, README that
  runs as written.
- ST-07 (#5) Evaluate reliability: loud failures, all 18 benchmarks load, one
  AdvBench scorer.
- ST-08 (#8) Demo notebooks rewritten on the final API, re-executed with real
  outputs.
- ST-09 (#6) Method fixes: disjoint BeaverTails splits, DeRTa token helper,
  depth-relative steer layers, SafeSwitch prober.
- ST-10 (#7) Integrate ST-06, ST-07 and ST-09 with cross-ticket fixes (DeRTa
  token, CLI dtype).
- ST-11 (#9) Method fixes (CAST, monitor, GradientAscent, ConstrainedSFT, DeRTa,
  steer) and paper-sized benchmark defaults.
- ST-12 (#10) Aya Vision and North support, `--safety-dataset`,
  `transformers>=5.15`.
- ST-13 Interop with the Lexsi stack: CuratorKIT dataset folders, CircuitKIT
  scores, `lexsi_provenance.json`, `push_to_hub`, version 0.1.6.
- Hackathon fixes (#18 and the PR stacked on it), below.

### Hackathon fixes (Cohere models)
- **`max_len` is sized from the chat template.** The QA data loaders default
  to `max_len=None`: `max(256, longest templated prompt + 256)`, capped at
  2048. Before, Tiny Aya's ~366-token preamble filled a fixed 256 and training
  ran on zero supervised tokens. An explicit `max_len` that leaves none raises.
- **Unlearn trains in fp32 for bf16 models too.** `upcast=True` (default)
  replaces `upcast_fp16`, which is a deprecated alias.
- **Refusal-direction sweep:** new `RefusalDirectionConfig.min_layer_fraction`
  (default 0.2): an early-layer winner, or none that lowers the refusal rate,
  falls back to the middle layer with a warning. Ties break toward the middle.
- **ReSta streams the safety vector one tensor at a time**; extra memory is a
  few fp32 copies of the largest tensor instead of about three of the model.
  New `device=` on `apply_resta` / `ReStaTrainer` (`"cpu"` keeps it off the GPU).
- **One BOS on vLLM text prompts and the suite's WildGuard judge.**
- **Batched generation left-pads** a passed-in right-padded tokenizer and
  restores its padding side afterwards.

### Interop (ST-13)
- **Datasets in.** `--train-dataset` / `--safety-dataset` (with the new
  `--train-config` / `--safety-config`) and every harden trainer's
  `train(...)` take a dataset folder plus config name, as
  `load_dataset(folder, config)` reads it (e.g. a CuratorKIT export and
  `sft_sharegpt`), as well as Hub ids, table names and single files. In
  Python: `trainer.train("./curated_out", dataset_config="sft_sharegpt")`, or a
  raw `datasets.Dataset`. SafeTune tokenises raw rows itself with the chat
  template: chat `messages`, ShareGPT `conversations`, Alpaca
  `instruction`/`input`/`output` (the `input` is kept) and prompt/response or
  DPO `prompt`/`chosen` columns, as strings or turn lists.
- **Prompt-only data is an error.** Rows without an assistant response
  (CuratorKIT `ppo` / `grpo`, a bare `text` column) used to be fine-tuned on
  as empty responses without a warning. Now such rows are skipped with a
  warning, and a dataset with none left raises `ValueError`.
- A dataset folder of several files no longer loads its first file silently
  when the split is not found; it raises and lists the files. Folders with a
  `README.md` (HF dataset folders) load with `load_dataset`, so
  `manifest.json`, `rejected.jsonl` and `lexsi_provenance.json` are never
  read as data.
- **CircuitKIT scores.** `load_circuit_info_from_file` / `get_circuit_info`
  read CircuitKIT `*_scores.json`: the `safety_units` / `layer_suggestions`
  keys CircuitKIT 0.2 writes, and files with only `node_scores`, from which
  the same keys are derived (`circuit_info_from_node_scores`, top 20% of
  nodes by default). A file with none of these keys raises `ValueError`; it
  used to return an empty `CircuitInfo`.
- **Provenance.** Every checkpoint (`save_checkpoint`: harden, recover,
  unlearn, `safetune patch --output`), results summaries directory and
  steering-vector directory gets a `lexsi_provenance.json`
  (`lexsi.provenance/1`), with the source model and the input datasets. When
  an input folder has its own `lexsi_provenance.json` it is embedded under
  `inputs[].provenance`, so lineage chains from CuratorKIT through SafeTune.
  `safetune.provenance` has the reader and writer.
- **Hub.** `safetune.push_to_hub(path, repo_id)` uploads a checkpoint folder
  (model, tokenizer, processor, provenance) or a results JSON with its
  provenance, creating the repo if needed.
- **Version.** 0.1.6. `safetune.__version__` comes from the installed package
  metadata; the release script and workflow no longer edit `__init__.py`.
  Deprecation messages that said "stops working in 0.2" now say 0.3.

### Changes to previously reported numbers

Every default changed since 0.1.3 that moves a number SafeTune reported
before, and the argument or `safetune.configure()` key that restores the old
behaviour. Explicit arguments win over `configure()`; the same keys work in the
`runtime:` / `datasets:` blocks of a `--config` YAML. None of these results
were re-run; reproduce an old number with its switch.

| What changed | Affected numbers | Restore the old behaviour |
|---|---|---|
| HarmBench is the paper's 400 text behaviours (standard 200 + contextual 100 + copyright 100); was standard only (200) | every HarmBench number: `evaluate()`, `trainer.evaluate()`, `load_bench_prompts`, `load_prompts` | `configure(datasets={"harmbench": {"config": "standard"}})` or `load_harmbench(subset="standard")` |
| WildJailbreak is the first 500 `adversarial_harmful` rows of the eval set (the selection `load_prompts()` has always used); `evaluate()` and `trainer.evaluate()` loaded all 2,210 rows, 210 of them benign | every WildJailbreak number outside `load_prompts()` | `configure(datasets={"wildjailbreak": {"where": None, "limit": None}})` |
| OR-Bench hard-1k (1,319 rows) and toxic (655) are reported as separate benchmarks: `orbench_overrefusal` and `orbench_toxic_refusal` in `trainer.evaluate()`, `orbench_hard` / `orbench_toxic` in `evaluate()` and steer evaluation; one combined `orbench_refusal` before | OR-Bench numbers; `safety_mean` | `configure(orbench_in_safety_mean=True)` |
| `safety_mean` averages the harm benchmarks only; OR-Bench (both splits) is out; before, one OR-Bench refusal rate over both splits was averaged in as "higher = safer" | every `safety_mean` / ρ | `configure(orbench_in_safety_mean=True)` |
| A bare `StringMatchJudge()` uses the 12-prefix scorer (`"prefix"`), as `trainer.evaluate()` always did for AdvBench; it used the 29-phrase GCG substring check | ASRT and Best-of-N attack success; any script with a bare `StringMatchJudge()` | `StringMatchJudge(mode="gcg")` or `configure(advbench_scorer="gcg")` |
| `run_judge("advbench")` drops `<think>...</think>` before matching, as `trainer.evaluate()` did | AdvBench scores of reasoning models via `run_judge` | none (the runner never scored think blocks) |
| `load_prompts("xstest" \| "jailbreakbench")` use `walledai/XSTest` and JBB-Behaviors `harmful`; were `natolambert/xstest-v2-copy` (gpt4) and `walledai/JailbreakBench` (100 harmful + 100 benign) | `load_prompts` numbers for these two | `configure(datasets={"xstest": {"source": "natolambert/xstest-v2-copy", "split": "gpt4"}})`; `configure(datasets={"jailbreakbench": {"source": "walledai/JailbreakBench", "split": "train"}})` |
| Evaluation raises when a benchmark fails; before, it was dropped from `safety_mean` silently | `safety_mean` of runs where a benchmark failed | `configure(eval_strict=False)` (the failure is logged and the benchmark still left out) |
| Steer default layers scale with depth (unchanged on 32 layers) | CAA, CAST, LinearProbeGuard, SafeSwitch, AlphaSteer on any model that is not 32 layers deep, e.g. Qwen2.5-0.5B (24), Llama-3.2-3B (28), Gemma-3-4B (34) | `configure(legacy_steer_layers=True)` or explicit layer arguments |
| SafeSwitch fits its prober in `calibrate`; before, it never fired | SafeSwitch on every model (old numbers equal the unsteered model's) | none: evaluate the unsteered model |
| TAR's default adversary set is disjoint from the harden contamination set | TAR without a `harm_dataset` | `configure(legacy_beavertails_splits=True)` |
| DeRTa's RTO token is "Sorry" under the model's tokenizer (19701 only on Llama-3) | DeRTa on every non-Llama-3 model | `rto_refusal_token_id=19701` |
| DeRTa trains as the authors do: the harmful prefix is masked, one RTO row per example (the harmful response, relabelled "Sorry"), one cross-entropy | DeRTa on every model | `DeRTaTrainer(legacy_derta=True)` or `configure(legacy_derta=True)` |
| ConstrainedSFT through the runner and CLI trains against its aligned reference; it was plain SFT | ConstrainedSFT (`--algo constrained`) on every model | `ConstrainedSFTTrainer(use_reference=False)` or `configure(legacy_constrained_sft=True)` |
| `GradientAscentTrainer` and `GradDiffTrainer` default `forget_clip=None` (TOFU's pure ascent); at 0.5 the forget term had no gradient on real data | GradientAscent and GradDiff (GradDiff trained on the retain set only) | `forget_clip=0.5` or `configure(legacy_ga_forget_clip=True)` |
| CAST fits its gate on chat-formatted prompts and gates each prompt of a batch; the gate never fired on chat models | CAST on every chat model (old numbers equal the unsteered model's when the gate never fired) | `CASTTrainer(chat_template=False, per_prompt_gate=False)` or `configure(legacy_cast_gate=True)` |
| AlphaSteer hooks each matrix at the layer it was fitted on (it fitted 10-19 and hooked 0-9) | AlphaSteer on every model | `AlphaSteerTrainer(legacy_alphasteer_layers=True)` or `configure(legacy_alphasteer_layers=True)` |
| ReSta's DARE drop rate is the paper's 0.3; it was 0.9 | ReSta with its defaults (DARE on) | `ReStaTrainer(dare_drop_rate=0.9)` or `configure(legacy_resta_drop_rate=True)` |
| `SpectralEntropyMonitor` leaves each prompt's first token (the attention sink) out and reads chat-formatted prompts | monitor flags and entropies | `SpectralMonitorConfig(skip_first_token=False, chat_template=False)` or `configure(legacy_spectral_monitor=True)` |
| Device and dtype are chosen per host (cuda > mps > cpu; bf16 where native, fp16 on pre-Ampere CUDA, fp32 on CPU); the CLI loads models in that dtype | runs on CPU, MPS or pre-Ampere CUDA; unchanged on Ampere+ CUDA | `configure(device=..., dtype="bfloat16")` |

Not number-changing: `eval_strict=True` and warnings for misspelled trainer
arguments stay the defaults, and MPS still defaults to bf16 (macOS 14+).
`CAAModel` hooks are on only inside `with` / `install()` and during its own
`generate()` / `__call__` (numbers through the wrapper are unchanged; call
`install()` after building it for the old always-on hooks), and AdaSteer
recomputes its coefficient per prompt also under `with` + `model.generate`
(its own `generate()` already did).

### Added
- **Runtime settings**: `safetune.configure(**settings)` / `safetune.get_config()`
  (and a `runtime:` block in `--config` YAML) set device, dtype, generation
  lengths, batch sizes, prompt caps, vLLM / lm-eval settings, judge settings,
  AdvBench refusal prefixes and LoRA defaults. Defaults are unchanged.
- **Dataset overrides**: every built-in dataset is resolved by short name from
  `safetune.data.dataset_ids`; point any of them at an HF id, a local
  `.jsonl`/`.json`/`.csv`/`.parquet` file or a URL with
  `configure(datasets={...})` or a `datasets:` YAML block.
- Results JSON records the effective runtime settings and dataset specs.
- `TransformersBackend` accepts a steering wrapper (`CASTModel`,
  `AdaSteerModel`, ...) as its model, so generation goes through the
  wrapper's own `generate()` (CAST's gate, AdaSteer's per-prompt coefficient).
- `examples/data/refusal_probes_demo.jsonl`: 8 demo rows in 4 languages for
  showing `configure(datasets=...)`.
- **Vision-language models** (Aya Vision, North / `cohere_compass`) load, train
  and save everywhere a model is loaded by path: the CLI, the runner, harden
  reference models, `evaluate()` and the quickstarts pick
  `AutoModelForImageTextToText` from the config. Every pillar works on the
  language model's decoder layers, default LoRA stays off the vision tower, and
  saved checkpoints keep the processor. Text-only data. `safetune[vision]` adds
  `torchvision`, which North needs.
- `safetune train --safety-dataset NAME [--safety-split SPLIT]`: the safety set
  for harden methods that take one (a `dataset_ids` name, HF id, local file or
  URL); methods without one exit with an error.

### Changed
- Requires `transformers>=5.15,<6` (North needs 5.15).
- The ten notebooks in `examples/notebooks/` use the one runner API,
  `configure()` and library helpers instead of copied prompt lists and
  helpers. Their committed outputs come from a CPU run, and the recover and
  monitoring notebooks use a real drifted checkpoint instead of noise.
- **One harden API**: `safetune.harden.<Name>Trainer` is now the same class as
  `safetune.runner.harden.<Name>Trainer` for every harden method. The
  `transformers.Trainer` subclasses that used to have those names are
  `<Name>HFTrainer` (DOOR: `SafetyDOORTrainer`). The old HF-style call
  (`<Name>Trainer(model=, args=, train_dataset=...)`) and the old submodule
  names still work until 0.2, with a `DeprecationWarning`.
- **Recover merge functions** (`task_arithmetic`, `somf_merge`,
  `learn_somf_mask`, `apply_resta`, `apply_lox`, `apply_lssf`,
  `apply_safemerge`, `apply_aaq`, `apply_safe_lora`) take everything after the
  model by keyword, so `base` and `aligned` can no longer be swapped by
  position. Old positional calls keep their old order until 0.2, with a
  `DeprecationWarning`.
- **Evaluation failures are loud.** `evaluate()`, `evaluate_with_vllm_backend()`
  and every trainer's `.evaluate()` raise when a benchmark fails to load or
  score (the error names the benchmark). `strict=False` or
  `configure(eval_strict=False)` records `{"error": ...}` for it and runs the
  rest. Unknown benchmark or judge names raise `ValueError` before anything
  loads. `safetune eval` exits 1 when any benchmark failed, and the evaluate
  quickstart no longer reports success after a failed step.
- **OR-Bench outside `safety_mean`.** OR-Bench hard-1k (over-refusal,
  `orbench_overrefusal`) and toxic (`orbench_toxic_refusal`) are reported on
  their own, as `orbench_hard` / `orbench_toxic` in `evaluate()`, and
  `safety_mean` averages the harm benchmarks only.
  `configure(orbench_in_safety_mean=True)` restores the old single
  `orbench_refusal` over both splits, averaged into the mean.
- **Benchmark sizes follow the SafeTune paper**: HarmBench 400 (standard +
  contextual + copyright), WildJailbreak 500 (the first 500
  `adversarial_harmful` rows). Each is a dataset-table entry you can override.
- **One AdvBench scorer.** `StringMatchJudge(mode="prefix" | "gcg")` is the only
  implementation. `trainer.evaluate()` and `run_judge("advbench")` use
  `configure(advbench_scorer=...)`, default `"prefix"` (what `trainer.evaluate()`
  always reported); `run_judge("advbench")` now also drops `<think>` blocks.
  A bare `StringMatchJudge()` (ASRT, Best-of-N) follows the same setting, so
  its default is `"prefix"` too; `mode="gcg"` keeps the 29-phrase check.
- `load_prompts("xstest" | "jailbreakbench")` use the same sources as
  `evaluate()` (`walledai/XSTest`, JBB-Behaviors harmful); RWKU loads the
  `forget_level2` QA probes.

### Removed
- The unused pack-runner HF dataset map (`load_pack_from_hf`,
  `run_safety_eval_from_hf` and the core eval CLI's `safety_eval` command).

### Fixed
- `evaluate()` can run every registered benchmark: `jailbreakbench`, `muse`,
  `rwku` and `safedialbench` no longer fail with `TypeError`, and `star1`,
  `mmlu` and `jailbreakbench` find their prompts.
- `sentencepiece` and `tiktoken` are declared; the default WildGuard judge
  needs them.
- CTRAP, SEAM, SEAL, ConstrainedSFT and DeRTa runner trainers now use their
  method kwargs; unknown trainer kwargs warn with the closest valid name.
- Device auto-selects cuda, then mps, then cpu; bf16 is used only where
  supported (fp16 on older CUDA, fp32 on CPU), so harden training no longer
  fails on CPU / T4, and lm-eval no longer hardcodes `cuda:0`.
- Judges fall back to transformers when vLLM is not installed.
- `import safetune.harden.lisa` (or any `safetune.<pillar>.<module>` path) no
  longer loads the module a second time with its own copies of every class.
- README: every Python block runs as written on CPU (except Evaluate, which
  needs a GPU for its judge); the Steer example uses `alpha=0.3` instead of the
  default 20, which turned output into noise on the README's 0.5B model.
- The TAR adversary fallback set no longer shares prompts with the harden
  contamination set (83 of 256 did on BeaverTails `30k_train`); only TAR's
  default data changes. `configure(legacy_beavertails_splits=True)` restores
  the old selection.
- Steer layer defaults (CAA, CAST, LinearProbeGuard, SafeSwitch, AlphaSteer)
  scale with model depth and are unchanged on 32-layer models. On shallower
  models CAA and CAST no longer return wrappers that change nothing, and
  LinearProbeGuard and AlphaSteer no longer fail; on 24/28/36-layer models the
  default layers move. `configure(legacy_steer_layers=True)` restores the
  absolute indices.
- `SafeSwitchTrainer.calibrate` fits its prober on the calibration prompts;
  before, the prober was never trained and the wrapper never fired.
- DeRTa's RTO transition token is the first token of "Sorry" under the
  model's tokenizer (19701 on Llama-3, the authors' id); before, every
  tokenizer got 19701, an unrelated token outside Llama-3.
  `rto_refusal_token_id=19701` restores the old id.
- CAST fits its gate on chat-formatted prompts, as generation sees them, and
  gates each prompt of a batch; the gate never fired on chat models. SafeSwitch
  fits its prober on chat-formatted prompts and scores every prompt of a batch
  (it scored the first). AlphaSteer hooks each matrix at the layer it was
  fitted on and runs on GPT-2.
- `CAAModel` no longer steers the model as soon as it is built, and still
  steers through its own `generate()` after a `with` block; AdaSteer no longer
  reuses the first prompt's coefficient under `with` + `model.generate`.
- `SpectralEntropyMonitor` leaves out the attention-sink token, which made the
  spectrum rank-1 and the entropy ~0 for every prompt, and reads chat-formatted
  prompts. Calibrated on the prompts it scans, it now flags a real safety drift
  (`examples/notebooks/safety_monitoring.ipynb`).
- DeRTa masks the harmful prefix and applies RTO to the harmful response only,
  in one cross-entropy, as the authors do; with trl 1.x its old RTO term had
  been silently skipped (`loss_type="chunked_nll"` returns no logits).
- ConstrainedSFT through the runner and CLI trains against its aligned
  reference (it was plain SFT).
- `GradientAscentTrainer` / `GradDiffTrainer` no longer clip the forget loss by
  default; the old 0.5 clip zeroed its gradient on real data.
- `ReStaTrainer` takes `dare_drop_rate`, default 0.3 (the RESTA paper's value;
  it was 0.9, which broke small models).
- `safetune train`, `patch` and `unlearn` load models in the runtime dtype
  (fp32 on CPU). transformers 5 loads bf16 by default, and CLI training on
  CPU ran at about 110 s per step.

## [0.1.3] - 2026-08-30

### Added
- **Docs**: EMNLP 2026 Demo tab with the accepted paper, walkthrough, and
  artifacts (`docs/emnlp-2026-demo/`).

### Changed
- **License**: updated to **LSAL v1.2** (see `LICENSE.md`). Academic research
  and teaching remain free on MIT-like terms. Organizations must now acknowledge
  their use to Lexsi Labs or obtain permission before internal evaluation,
  auditing, or use on their own models (Section 1A). Commercial use still
  requires a separate license. The patent clause now reserves all patent rights
  (Clear BSD style) instead of granting a noncommercial patent license. New
  clauses: users bear responsibility for their own use and deployment decisions
  (Section 7, with an indemnity limited to organizational and commercial use),
  third-party base-model and benchmark licenses continue to apply (Section 4A),
  modified redistributions must be marked as modified (Section 2), and
  survival, severability, and version-applicability terms (Sections 8 and 9).
- **License**: SafeTune is now released under the **Lexsi Labs Source Available
  License (LSAL) v1.1** (see `LICENSE.md`) — an MIT-style grant restricted to
  noncommercial purposes, with a Responsible Use clause barring production
  deployment of unrepaired drifted checkpoints. Earlier changelog entries
  referring to the MIT License describe pre-LSAL releases. Commercial
  licensing: support@lexsi.ai.

## [0.1.1] - 2026-06-29

### Added

- **`safetune.config.SafeTuneConfig`** — declarative YAML config for CLI runs.
  `SafeTuneConfig.from_yaml(path)` loads all standard flags plus
  method-specific hyperparameters (any unknown key lands in `method_kwargs`
  and is forwarded directly to the trainer constructor).
- **`--config` CLI flag** — `safetune train --config run.yaml` injects YAML
  values as parser defaults; explicit CLI flags still take precedence.
- **`--train-dataset` / `--train-split` CLI flags** — training dataset is now
  fully configurable. `--train-dataset beavertails` (default) or any HF
  dataset id; `--train-split 30k_train` (default) or any split name.
- **`safetune.runner._registry`** — centralised algo registry replacing the
  inline dicts in `cli.py`. `register_harden()`, `register_recover()`, and
  `register_unlearn()` let third-party code extend the method menu at runtime
  without editing library files.
- **`SaLoRATrainer`** — `lora_alpha`, `lora_dropout`, and `target_modules`
  are now configurable constructor kwargs (previously hardcoded inside `train()`).
- **`_RecoverBase.apply()` contract** — method is now documented: return the
  patched model, never write to disk, accept method-specific keyword overrides.
- **Dev runbook §6** — end-to-end guide for adding a new method: implement
  trainer → re-export → add one registry entry.

### Changed

- `cli.py` now imports algo registries from `runner/_registry.py` instead of
  maintaining inline dicts.  `safetune list` output is unchanged.
- `--no-peft` flag removed from the CLI (it was registered but never consumed).

### Fixed

- `_derive_model_id` duplicated across `_HardenBase` and `_RecoverBase` —
  consolidated into `model_utils.derive_model_id()`.
- Dead `R = None` / `_ensure_recover_imports()` globals removed from
  `runner/recover/_base.py` (subclass files already imported directly).
- Stray multi-line bug-fix comment removed from `LoXHardenTrainer.train()`.

## [0.1.0] - 2026-06-24

**First public release.** SafeTune is a library of ~100 alternative LLM-safety
methods — train-time hardening, weight-space recovery and unlearning,
inference-time steering, plus diagnosis and evaluation — for the Hugging Face
ecosystem. Every method is faithfulness-audited against its cited paper.

Validated on an NVIDIA L40S with torch 2.8 / transformers 5.12 / trl 1.6
(full suite: 354 passed, 4 skipped). Docs build clean under `mkdocs --strict`.

### Added — the library

- **~100 methods across 4 intervention pillars + 2 instrumentation tools:**
  - **Harden** (26 methods): `SafeGradTrainer`, `LisaTrainer`, `SurgeryTrainer`,
    `AntibodyTrainer`, `LookAheadTrainer`, `vaccine_loss`, `tar_outer_loss`,
    and 19 more across 8 mechanism families.
  - **Recover** (24 methods): `apply_resta`, `apply_lox`, `apply_safemerge`,
    `apply_ctheta`, `task_arithmetic`, and 19 more across 6 granularities
    (whole-model → subspace → layer → neuron → circuit).
  - **Unlearn** (6 methods): `rmu_unlearn`, `npo_unlearn`,
    `gradient_ascent_unlearn` (plus its GradDiff variant), `flat_unlearn`,
    `simdpo_unlearn`.
  - **Steer** (19 methods): `RefusalDirectionModel`, `AdaSteerModel`,
    `SafeSwitchModel`, `AlphaSteerModel`, `SafeSteerModel`, and 14 more.
    Includes `steer.run(...)` with `hf` / `vllm-hook` / `vllm-logits` backends.
  - **Interpret** (6 methods): `identify_safety_neurons`, `safety_circuit_info`,
    `eap_safety_circuit`, `CircuitInfo` (round-trippable JSON/YAML).
  - **Evaluate** (24 methods): `evaluate(...)` with HarmBench, XSTest, AdvBench,
    WildJailbreak, and more. `BoNAttack`, `AbliterationAttack`, WildGuard /
    LlamaGuard-3 judges.
- **Faithfulness audit**: every method compared against its cited paper,
  corrected where it diverged, and labelled. 100 faithful, 1 simplified,
  5 SafeTune variants, 0 broken. Per-method verdicts with `file:line` evidence
  in the [Feature Map](docs/reference/feature-map.md).
- **Quickstart demos**: `quickstart.py` (steer), `recover_quickstart.py`,
  `harden_quickstart.py` — all run on `Qwen/Qwen2.5-0.5B-Instruct`, no GPU
  required.
- **Colab notebooks**: 4 interactive notebooks (steer, recover, harden,
  unlearn) mirroring the quickstart scripts.
- **CLI**: `safetune` / `st` commands dispatch to real pillar APIs (harden,
  evaluate, recover).

### Added — documentation

- **MkDocs Material site** with purple/amber design system, dark mode, sticky
  tabs, instant navigation, search, code copy, Mermaid diagrams.
- **Doc structure**: `getting-started/` (install, quickstart, taxonomy),
  `guides/` (6 method-group guides — 4 intervention pillars + 2 instrumentation
  tools), `trust/` (feature map, results, scope, audit details), `reference/`
  (paper table, eval protocols), `tutorials/` (Colab hub), `community/` (FAQ,
  contributing, changelog), `blog/`.
- **Landing page** pitching 4 intervention pillars and 2 instrumentation tools,
  with real impact numbers, a decision table, scenario-based usage examples,
  and runnable code tabs.
- **Lexsi Labs branding**: compass logo, Lexsi logo footer (dark/light
  variants).

### Added — standard library files

- `CITATION.cff` — citation metadata.
- `.github/ISSUE_TEMPLATE/` — bug report, faithfulness report, feature request.
- `.github/pull_request_template.md`.
- `.github/workflows/docs.yml` — GitHub Pages docs deploy.
- `.github/workflows/smoke.yml` — CI smoke test.
- `LICENSE` — MIT, 2026.

### Changed (Major)

- **CLI rewritten** to dispatch to harden / evaluate / recover pillar APIs
  instead of the old SFT/DPO/PPO/GRPO orchestrator stubs.
- **`verify` → `evaluate` rename**: `safetune.evaluate` is the current name;
  `safetune.verify` was later removed in v0.1.0.
- **Recover uniform input contract**: every `apply_*` accepts `target=`
  (`model=` / `finetuned=` kept as aliases).
- **2-tier, input-keyed taxonomy**: Tier 1 Interventions (harden / recover /
  unlearn / steer), Tier 2 Instrumentation (interpret / evaluate).
- **Benchmark menu** categorized into jailbreak / over_refusal / capability /
  domain / tamper.

### Fixed

- **FLAT unlearning rewritten** to the faithful f-divergence loss
  (Wang et al., ICLR 2025, arXiv:2410.11143).
- **Decoding steer methods** (SafeDecoding, ContrastiveDecoding, ProxyTuning,
  Nudging): logit width reconciliation for padded `lm_head`s.
- **SafeGradTrainer**: whole-model gradient dot overflow on >2B params.
- **EAP-IG**: integrated-gradient interpolation off-by-one.
- **DOORTrainer**: DPO + DOOR hybrid loss replaced with paper-faithful DOOR-only.
- **Import-order fragility**: all 12 Harden configs guarded against
  transformers/trl import race.
- **Packaging**: MIT license classifier (was Proprietary); upper version bounds;
  guarded trl imports for 1.x compatibility.

### Removed

- ~3.8 GB of raw per-prompt eval generations and internal scratch/planning docs.
- 12 broken in-house attack reimplementations (GCG, PAIR, TAP, AutoDAN, …) —
  not faithful to their papers.
- Orphaned pipeline-orchestration subsystem (`core/options.py`,
  `core/orchestrator.py`, `core/callbacks.py`).
- Stale duplicate docs (`docs/safetune-docs/`, `docs/archive/`).
- `requirements.txt` (consolidated into `pyproject.toml`).

## [Legacy numbering]

Versions before the June 2026 renumbering: this line peaked at 0.6.0
(`finetunehub` → SafeTune era) and was reset to 0.1.0 when the public
PyPI releases started. Kept as history; none of it is on the index, and
`scripts/check_version_consistency.py` does not count it as releases.

### [0.6.0] - 2026-05-16

### Changed (Major)
- **Taxonomy overhaul**: replaced the flat "Four/Five Pillars" list with a
  **2-tier, input-keyed taxonomy**. Tier 1 · Interventions — Train-time
  (`harden`), Weight-space (`recover` + `unlearn`), Inference-time (`steer`);
  Tier 2 · Instrumentation — Diagnose (`interpret`), Measure (`evaluate`).
  SafeTune is framed as a **library of alternative methods, not a pipeline**.
- **`verify` → `evaluate` rename**: package dir `src/safetune/verify/` →
  `evaluate/` (`verify/eval/` → `evaluate/suite/`). `safetune.verify` was a
  back-compat alias that emits a `DeprecationWarning`.
- `interpret` and `unlearn` are now first-class importable submodules
  (`import safetune.interpret`, `import safetune.unlearn`).
- **Repo layout**: `emnlp_exp/` → `experiments/emnlp2026/`, `paper/` →
  `experiments/paper/`, `safetune_check/` → `audit/`, validation scripts →
  `tests/support/`. Top level is now deliverable directories only.

### Added
- **Gold-standard re-check pass**: cloned 12 upstream reference repos into
  `audit/reference_repos/` and diffed every 🟡 SafeTune adapter against the
  originating code. Result: 12 methods upgraded 🟡→✅ (antidote, pke, safereact,
  tracin_influence, DOORTrainer, LookAheadTrainer, tar_outer_loss, AdaSteerModel,
  RRFAEnsemble, eap_safety_circuit, task_arithmetic, NudgingProcessor). After a
  final 🟡→✅ sweep the audited surface is ✅ 89, 🟡 0, 🟠 2. Two bugs found and
  fixed in the same pass (see below).
- **`docs/REFERENCES.md`** — per-method table covering all ~91 audited methods: paper,
  venue, arXiv, official repo link, 1-sentence description, non-obvious inputs
  required, outputs, and faithfulness badge.
- **`safetune.steer.run(model, backend=...)`** — one steering-generation entry
  point with `hf` / `vllm-hook` / `vllm-logits` backends. The vLLM adapters are
  promoted into `safetune.steer.backends`.
- **Recover uniform input contract** — every `recover` `apply_*` accepts the
  canonical `target=` keyword (`model=` / `finetuned=` kept as aliases).
- **Benchmark menu** — `evaluate.suite` registry categorized into
  jailbreak / over_refusal / capability / domain / tamper, with
  `list_benchmarks()` and `benchmarks_by_category()`.
- **`docs/LIBRARY_CHECKLIST.csv`** — the `FEATURE_MAP.md` audit map flattened to
  one CSV row per method across every pillar (~100 rows: Pillar, Method,
  Entry-point file, Runs?, Faithful?, Audit status). Regenerated from
  `FEATURE_MAP.md` by `scripts/gen_library_checklist.py`.

### Removed
- **Orphaned pipeline-orchestration subsystem**: `core/options.py`,
  `core/orchestrator.py`, `core/callbacks.py` and the `adversarial` /
  `constitutional` / `lifelong` / `multiturn` / `selfplay` config modules —
  pipeline framing with no executor.
- Retired the broken/undefined entry points `pipelines.pipeline` and
  `training.orchestrator.run_sft/dpo/ppo/grpo` from the public surface.

### Fixed
- `harden/safegrad.py` — `SafeGradTrainer` global gradient surgery called
  `torch.dot` on the concatenated whole-model gradient, overflowing the BLAS
  int32 length bound for any model above ~2B parameters; replaced with the
  numerically identical `(a*b).sum()`.
- `harden/lisa.py` — `LisaTrainer` crashed with `'DataLoader' object is not
  subscriptable` when `alignment_dataset` was passed as an already-built
  `DataLoader`; it is now tolerated.
- Import-order fragility across the whole `harden` pillar — every Trainer
  config was declared as `@dataclass class XConfig(Base if _IMPORT_ERROR is
  None else object)`. When `transformers`/`trl` was imported *before*
  `safetune` (a very common order), the backend guard had not run, the nested
  import failed, the base silently degraded to `object`, and `@dataclass`
  regenerated `__init__` from only the local fields — so e.g.
  `SafeGradConfig(output_dir=...)` raised `unexpected keyword argument`. Fixed
  in all 12 affected configs (`SafeGradConfig`, `LisaConfig`, `AsFTConfig`,
  `SPPFTConfig`, `STARDSSConfig`, `SAPConfig`, `SurgeryConfig`, `CSTConfig`,
  `DOORConfig`, `DeRTaConfig`, and `core.optim.AntibodyConfig`) by guarding the
  `@dataclass` declaration behind the import check, with an explicit stub in
  the failure branch instead of a misleading half-built dataclass.
- `eval/cli.py` — imported `EvaluationRegistry` from `eval/registry.py` (which
  defines only the *metric* registry `EvalRegistry`); the CLI actually uses the
  *task* registry's `get_task` / `list_tasks` API, which lives in `eval/core.py`.
  Repointed the import to `core.EvalRegistry`; the module now imports and the
  whole package is import-clean (247/247 modules).
- `docs/FEATURE_MAP.md` audit badges synced with the `fix/*_RESOLVED.md`
  fixlogs, which were never reflected in the map. `apply_mscp` and `apply_lssf`
  move 🟠→🟡 — both cite verified papers (arXiv:2508.09190; arXiv:2602.00038,
  ACL 2025) and faithfully implement their projection equations; the stale
  "un-cited" notes were wrong. `apply_deeprefusal` and `ASRTCallback` stay 🟠
  (honest SafeTune-original heuristics, no faithful published equivalent) with
  corrected notes. Post-fix distribution is now 89✅ / 0🟡 / 2🟠 / 0🔴 / 0⚫
  over the 91 audited components.
- **`core/interpret/eap.py`** — EAP-IG integrated-gradients interpolation
  loop started at `k=1` (`range(1, steps+1)`), causing gradient sampling to
  include the clean endpoint and exclude the corrupted endpoint. Upstream
  EAP-IG (`hannamw/EAP-IG`) samples `{0/steps, ..., (steps-1)/steps}`;
  fixed to `range(0, steps)` to match.
- **`harden/door.py`** — `SafetyDOORTrainer` was mixing DPO + DOOR loss
  (calling `super().compute_loss()` then adding the DOOR term). The paper's
  reference implementation uses DOOR alone (`gd_npo_loss` on a plain `Trainer`).
  Fixed: `DOORConfig.door_pure_mode=True` (default) now returns only the DOOR
  term; `door_pure_mode=False` preserves the old hybrid for back-compat.

### [0.5.0] - 2026-04-12

### Changed (Major)
- **Architectural Overhaul**: Transitioned to the "Four Pillars of Safety" taxonomy: **Recover**, **Harden**, **Steer**, and **Verify**.
- **Package Modernization**: Renamed internal package to `safetune` (lowercase) for standard Python convention.
- **Modular Rewards**: Decomposed the monolithic `rewards/core.py` into categorical sub-modules (`text`, `nlp`, `safety`, `code`, `math`, `specialized`).
- **Pipelines API**: Introduced high-level unified `pipelines` API for simplified safety orchestration.
- **CLI Refactor**: Separated CLI logic into `cli.py` and extracted training orchestration into `training/orchestrator.py`.
- **Global Branding**: Standardized all references to **SafeTune** across documentation and code.

### Added
- **Specialized Rewards**: New reward functions for Medical, Legal, and Financial domains.
- **Unified Configuration**: Streamlined `UnifiedSafetyConfig` supporting the 4-pillar structure.
- **Project Structure**: Cleaned up the root directory and standardized the `src/` layout.

### Removed
- Legacy monolithic files: `src/safetune/main.py` and `src/safetune/rewards/core.py`.
- Redundant backup files and scripts.

### [0.2.0] - 2026-01-18

### Changed (Major)
- **Renamed library from `finetunehub` to `SafeTune`**
  - Package directory: `src/finetunehub/` → `src/safetune/`
  - All imports updated: `from finetunehub.*` → `from safetune.*`
  - CLI commands: `safetune` (primary), `at` (short alias)
  - Entry points and pyproject.toml fully updated
  - All documentation, examples, and tests migrated

- **License**: SafeTune is released under the **MIT License** (see `LICENSE`).
  - An earlier source-available license proposal was *not* retained; the
    project is MIT-licensed — free for research, academic, commercial and
    personal use.

### Added
- **Comprehensive Documentation System** (59 markdown files):
  - Complete documentation structure integrated into main project
  - Getting Started guides (5 docs): Installation, Quick Start, Basic Concepts, Configuration, Backend Selection
  - User Guide (9 docs): Overview, SFT, RL, Evaluation, Reward Functions, Model Management, Sample Logging, Troubleshooting
  - Algorithm Documentation (10 docs): DPO, PPO, GRPO, GSPO, DAPO, Dr. GRPO, GBMPO, Counterfactual GRPO, BOLT with detailed explanations
  - Backend Documentation (4 docs): Overview, TRL Backend, Unsloth Backend, Comparison
  - API Reference (6 docs): Complete API documentation with all parameters
  - Examples (4 categories): Overview, SFT, RL, Advanced examples
  - Advanced Topics (4 docs): Architecture, Custom Backends, Distributed Training, Performance
  - Contributing guides (3 docs): Guide, Code Style, Testing
  - Notebooks section for interactive tutorials

- **Enhanced Documentation Infrastructure**:
  - MkDocs configuration with Cinder theme
  - Mermaid diagram support for architecture visualization
  - Jupyter notebook integration
  - Automatic API documentation generation with mkdocstrings
  - Custom HTML overrides and styling
  - Logo and branding assets removed (migrated to SafeTune branding)

### Removed (Code Cleanup)
- **Deleted unused/backup directories and files** (Total: ~7,500 lines removed):
  - `src/finetunehub/eval_old/` - Old evaluation framework (7 files, 2,913 lines)
  - `src/finetunehub/backends/trl/rl/ppo/ppo_old.py` - Old PPO implementation (1,437 lines)
  - `src/finetunehub/backends/trl/rl/grpo/grpo_old.py` - Old GRPO implementation (1,264 lines)
  - `src/finetunehub/backends/unsloth/rl/ppo/ppo_old.py` - Old Unsloth PPO (1,833 lines)
  - `src/finetunehub/cli_commands/` - Unused CLI modules
  - `src/finetunehub/cli/unified-old.py` - Old CLI backup
  - `src/finetunehub/rl/` and `src/finetunehub/sft/` - Backward compatibility wrappers
  - `src/finetunehub/scripts/` - Directory removed (contents moved)
- **Removed old `finetunehub` package directory** - Fully replaced by `safetune`

### Changed
- **Reorganized BOLT utilities**:
  - Moved `precompute_baseline.py` from `src/finetunehub/scripts/` → `examples/bolt_training/`
  - Rationale: Co-locate BOLT-specific utilities with BOLT examples for better discoverability

### Breaking Changes
- **Library renamed**: All imports must change from `finetunehub` to `safetune`
  - `from finetunehub.core.rl import *` → `from safetune.core.rl import *`
  - `from finetunehub.eval import *` → `from safetune.eval import *`
  - CLI: `finetunehub train` → `safetune train`
- **Removed backward compatibility wrappers**: The deprecated import paths have been removed
  - All examples and documentation have been updated to use the new paths
  - All test files have been migrated to the new import structure

### [0.1.0] - 2026-01-18

### Added
- Initial release with comprehensive GRPO support
- Backend support for both TRL and Unsloth
- GSM8K math reasoning examples
- Multiple bug fixes and improvements
