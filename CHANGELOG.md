# Changelog

All notable changes to SafeTune are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **License**: SafeTune is now released under the **Lexsi Labs Source Available
  License (LSAL) v1.1** (see `LICENSE.md`) — an MIT-style grant restricted to
  noncommercial purposes, with a Responsible Use clause barring production
  deployment of unrepaired drifted checkpoints. Earlier changelog entries
  referring to the MIT License describe pre-LSAL releases. Commercial
  licensing: support@lexsi.ai.

## [1.0.1] - 2026-06-29

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

## [1.0.0] - 2026-06-24

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
  in the [Feature Map](docs/trust/feature-map.md).
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
  `safetune.verify` was later removed in v1.0.0.
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

## [0.6.0] - 2026-05-16

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

## [0.5.0] - 2026-04-12

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

## [0.2.0] - 2026-01-18

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

## [0.1.0] - 2026-01-18

### Added
- Initial release with comprehensive GRPO support
- Backend support for both TRL and Unsloth
- GSM8K math reasoning examples
- Multiple bug fixes and improvements
