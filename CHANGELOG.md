# Changelog

All notable changes to SafeTune are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-07-11

**First public release.** SafeTune is a library of LLM-safety methods —
train-time hardening, weight-space recovery and unlearning, inference-time
steering, plus diagnosis and evaluation — for the Hugging Face ecosystem. It
is organized into a 2-tier taxonomy: Tier 1 Interventions (Harden / Recover /
Unlearn / Steer) and Tier 2 Instrumentation (Interpret / Evaluate).

- Every method is faithfulness-audited against its cited paper. Per-method
  verdicts (Faithful / Simplified / Variant) with evidence are in the
  [Feature Map](docs/reference/feature-map.md); see
  [Scope & Limitations](docs/community/scope.md) for what each verdict means
  and which methods must not be cited as their named paper's method.
- **CLI**: `safetune` / `st` commands dispatch to the harden, recover, and
  evaluate pillar APIs.
- **Quickstart demos and Colab notebooks** covering each pillar, runnable on
  a small open model with no GPU required.
- **MkDocs Material documentation site**: getting-started, per-pillar user
  guides, trust/scope docs, API reference.
- Licensed under the **Lexsi Labs Source Available License (LSAL) v1.1** (see
  [LICENSE.md](LICENSE.md)) — noncommercial use is free; commercial licensing
  is available from support@lexsi.ai.
