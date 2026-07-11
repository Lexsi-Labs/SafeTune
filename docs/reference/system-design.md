# SafeTune — System Design Document

## 1. Product Blueprint & Architecture

### 1.1 Overview

SafeTune is a PyPI library of ~100 LLM safety methods organized into a
2-tier, input-keyed taxonomy. It is **not a pipeline** — each task has many
independent methods that solve it by different mechanisms; users pick one per task.

### 1.2 Package Architecture

```
safetune/
├── interventions/          Tier 1 — methods that CHANGE safety
│   ├── harden/             train-time defenses (SafeGrad, Lisa, Vaccine, etc.)
│   ├── recover/            weight-space patching (RESTA, C-Θ, LoX, etc.)
│   ├── unlearn/            forget-set training (RMU, NPO, FLAT, etc.)
│   └── steer/              inference-time steering (CAA, RefusalDir, etc.)
│
├── instrumentation/        Tier 2 — methods that OBSERVE safety
│   ├── interpret/          diagnose: safety neurons, circuits, EAP
│   └── evaluate/           measure: redteam attacks, benchmarks, judges
│
├── core/                   shared building blocks
│   ├── optim/              optimizer-level safety wrappers
│   ├── patches/            weight-patching primitives
│   ├── circuit_kit/        CircuitInfo data structure
│   ├── repeng/             representation engineering tools
│   ├── data_compiler/      data compilation pipeline
│   ├── eval/               low-level evaluation infrastructure
│   ├── runtime/            inference-time hook runtime
│   ├── extras/             steering vectors, safety subspace
│   └── safety/             multi-turn safety (attack + defense)
│
├── runner/                 high-level Trainer API
│   ├── harden/             per-method trainers (safegrad, lisa, vaccine, etc.)
│   ├── recover/            per-method trainers (ctheta, resta, somf, etc.)
│   ├── steer/              per-method trainers (refusal_dir, caa, cast, etc.)
│   └── unlearn/            per-method trainers (rmu, npo, flat, etc.)
│
├── data/                   data loaders (harmbench, beavertails, advbench, etc.)
├── rewards/                reward function factory
├── utils/                  logging, auth, device management, checkpointing
└── __init__.py             clean imports with sys.modules aliases
```

### 1.3 Execution Layers

| Layer | Technology | Hardware | Function |
|---|---|---|---|
| **Method Logic** | PyTorch, Transformers | GPU (CUDA/MPS) | Weight transforms, training loops, activation hooks |
| **Evaluation** | vLLM, HF Generate | GPU (vLLM for throughput) | Run safety benchmarks, score generations |
| **Data Pipeline** | HuggingFace Datasets | CPU (memory-mapped) | Load, compile, tokenize safety datasets |

### 1.4 Key Design Decisions

1. **No pipeline orchestration.** Users pick one method per task, not a sequence.
   This prevents accidental method entanglement and keeps each method independently
   auditable.
2. **Lazy imports for heavy optional backends.** `vllm` (and the interpret
   extras) are imported only when a method that needs them runs, so the base
   install does not pull them in. Note that `transformers`, `trl`, `torch`,
   `datasets`, and `peft` are eager dependencies imported at `import safetune`
   time, so a cold `import safetune` still takes a few seconds.
3. **Faithfulness audit built-in.** Every method carries an audit badge
   (Faithful / Simplified / Variant / Wrong / Stub). Only Faithful methods may be
   cited as the named paper.
4. **Runner layer as optional wizard.** The `runner/` package wraps pillar methods
   in a `Trainer().train().eval().save_results()` workflow. Users can also use
   the pillar packages directly (`safetune.recover.apply_resta(...)`).

### 1.5 Compute Cost Isolation

| Operation | Hardware | Estimated Cost |
|---|---|---|
| Training (Harden/Unlearn) | 1× GPU (A100 80GB) | $2-5/hr |
| Weight-patching (Recover) | 1× GPU (A100 80GB) | $0.5-2/hr |
| Inference steering (Steer) | 1× GPU (A100 80GB) | $0.5-1/hr |
| Evaluation safety | 1× GPU (vLLM) | $0.5-2/hr per 1000 prompts |
| Evaluation utility | 1× GPU (lm-eval) | $1-3/hr per benchmark suite |
| Data compilation | CPU (8+ cores) | <$0.10/hr |

### 1.6 Security & Isolation

- No hardcoded API keys / tokens in source
- HuggingFace authentication via `huggingface-cli login` or `HF_TOKEN` env var
- Model weights downloaded to `~/.cache/huggingface/` (not bundled)
- SafeTune disables TF/Flax backends on import to prevent transitive import errors
