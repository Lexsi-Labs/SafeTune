# How to use these docs

The SafeTune documentation site is organized to match the library's
[2-tier taxonomy](taxonomy.md). Here is how the UI works and how to find what
you need.

## Navigation structure

```
Getting Started ──→ Install · quickstart · core concepts (taxonomy)
  │
  ├─ User Guide ───→ Four intervention pillars (harden, recover, unlearn, steer)
  │                  + two instrumentation tools (interpret, evaluate)
  ├─ Reference ────→ API reference · CLI reference · references (papers)
  │                  · feature map · system design · API contract
  ├─ Examples ─────→ Python scripts · notebooks · case study
  └─ Community ────→ FAQ · contributing · scope · changelog
```

### Method group guide pages

Each of the six method groups has a guide page like this:

```
User Guide → Harden
  ├── Overview page      What this pillar does, when to use it
  ├── Gradient surgery   SafeGrad · PlainSFT
  ├── Data shaping       Lisa · SPPFT · LookAhead · STARDSS · DeRTa · CST
  ├── Regularization     AsFT · SAP · Surgery · Booster
  ├── Representation     Vaccine · T-Vaccine
  ├── Tamper-resistant   RepNoise · CTRAP · SEAM · DOOR · MART · ...
  ├── Data selection     SEAL · ConstrainedSFT
  ├── Pre-FT hardening   LoX-Harden
  └── TAR routing        TAR
```

The left sidebar shows the full tree. Click any method name to see its
dedicated page with:
- **Input/output contract** — what you give it, what you get back
- **Quick example** — a runnable code snippet
- **Parameters** — every config field with its default
- **When to use** — which scenarios this method suits
- **Trade-offs** — capability cost, speed, memory

## Features

### Instant search

Press `s` or `/` or click the search box (top bar). Search is client-side and
works offline. Results group by page with highlighted matches.

### Dark / light mode

Click the sun/moon icon in the top bar. Your preference is remembered across
sessions.

### Copy code

Every code block has a clipboard button (top-right corner). Click to copy the
snippet, then paste into your terminal or editor.

### Edit this page

Every page has an "Edit" link in the top-right action bar. Click it to open the
markdown source on GitHub and submit a documentation fix.

### Version selector

If multiple versions exist, use the version dropdown (top bar) to switch
between release docs.

## The taxonomy: how methods are organized

```
TIER 1 — INTERVENTIONS  (methods that CHANGE a model's safety)
├── Train-time           harden      — replace your SFT trainer
├── Weight-space         recover     — patch a finished model (no training)
│                        unlearn     — train a model to forget a capability
└── Inference-time       steer       — wrap a frozen model with hooks

TIER 2 — INSTRUMENTATION (methods that OBSERVE safety)
├── Diagnose             interpret   — locate directions / neurons / circuits
└── Measure              evaluate    — run redteam attacks + judge scoring
```

Each box is a **catalog of alternatives**. You pick one method per task,
not a sequence.

## Audit badges

Every method in the [Feature Map](../reference/feature-map.md) carries a badge:

| Badge | Meaning | Can I cite the paper? |
|---|---|---|
| **Faithful** | Implements the cited algorithm exactly | Yes |
| **Simplified** | Reduced but algorithmically correct | With caveats |
| **Variant** | SafeTune heuristic; not the named algorithm | No (cite as "SafeTune variant") |
| **Wrong** | Algorithm is buggy or incorrect | Do not use |
| **Stub** | Not yet implemented | — |

The [Scope & Audit Status](../community/scope.md) page explains the full audit
methodology. Only Faithful methods may be cited as the named method from
their paper.

## Keyboard shortcuts

| Key | Action |
|---|---|
| `s` or `/` | Open search |
| `p` or `,` | Previous page |
| `n` or `.` | Next page |
| `Esc` | Close dialog / clear search |
| `1`–`4` | Switch between tabs (in tabbed content) |

## Finding what you need

| If you want to… | Go to… |
|---|---|
| Install the library | [Getting Started → Installation](installation.md) |
| Understand the taxonomy | [Getting Started → Core Concepts](taxonomy.md) |
| Pick a harden method | [User Guide → Harden](../user-guide/harden.md) |
| See if a method is trustworthy | [Reference → Feature Map](../reference/feature-map.md) |
| Run a quick demo | [Examples → Notebooks](../examples/notebooks.md) |
| See all method papers | [Reference → References](../reference/references.md) |
| Contribute a fix | [Community → Contributing](../community/contributing.md) |
| Understand system internals | [reference/system-design.md](../reference/system-design.md) (dev docs) |
