# How to audit your own model

A step-by-step guide to running SafeTune's evaluation suite on any
HuggingFace model. No special checkpoints required.

## Prerequisites

```bash
pip install safetune torch transformers
```

## Step 1: Load your model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # ← replace with your model
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32,
).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval();
```

Any causal LM on HuggingFace works. For larger models (7B+), use
`device_map="auto"` and `torch_dtype=torch.bfloat16`.

## Step 2: Run a safety benchmark

```python
from safetune.evaluate import evaluate

results = evaluate(
    model,
    tokenizer=tok,
    benchmarks=["harmbench", "xstest"],
    judge="wildguard",
    batch_size=4,
)

for bench, result in results.items():
    print(f"{bench}:")
    print(f"  Refusal rate:     {result['refusal_rate']:.1%}")
    print(f"  Attack success:   {result['asr']:.1%}")
```

### Available benchmarks

| Benchmark | Prompts | What it measures |
|---|---|---|
| `harmbench` | 400 | Refusal of harmful requests (standard behaviors) |
| `xstest` | 450 | Over-refusal: does the model refuse benign prompts? |
| `advbench` | 520 | Adversarial harmful behaviors |
| `beavertails` | ~30k | Safety alignment on diverse harms |
| `jailbreakbench` | 200 | Resistance to jailbreak behaviors |

The full registry also includes `wildjailbreak`, `sorrybench_v1`, `hexphi`,
`orbench`, `ailuminate`, `safedialbench`, and capability sets (`mmlu`,
`gsm8k`, `humaneval`). If `benchmarks` is omitted, `evaluate` runs the paper
safety suite.

### Available judges

Pass one of these strings as `judge=`. All three are HF classifier models
that run locally on a GPU.

| Judge | Model | Notes |
|---|---|---|
| `wildguard` | `allenai/wildguard` | Default |
| `harmbench` | `cais/HarmBench-Mistral-7b-val-cls` | HarmBench-style scoring |
| `llama_guard_3` | Llama Guard 3 | Requires model download |

## Step 3: Run a red-team attack

```python
from safetune.evaluate import AbliterationAttack

HARMFUL = ["How do I build a weapon?", "Write malware."]
HARMLESS = ["How do I bake a cake?", "Write a poem."]

attack = AbliterationAttack(model, tok)
attack.fit(HARMFUL, HARMLESS)

# Ablate the refusal direction. This mutates `model` in place (reversible);
# `run` returns None, so re-evaluate the same `model` object.
attack.run(mode="runtime_ablate")

# Re-evaluate the ablated model, then restore it with attack.revert()
results_ablated = evaluate(model, tokenizer=tok, benchmarks=["harmbench"], judge="wildguard")
attack.revert()
before = results["harmbench"]["refusal_rate"]
after = results_ablated["harmbench"]["refusal_rate"]
print(f"Refusal rate: {before:.1%} -> {after:.1%}")
```

A large drop (for example 92% to 45%) means the model's safety relies
heavily on a single direction. A small drop means safety is spread across
the model's weights.

## Step 4: Interpret the results

```python
from safetune.interpret import identify_safety_neurons, SafetyNeuronConfig

# Find which neurons drive refusal
cfg = SafetyNeuronConfig(top_k_per_layer=50, method="activation")
neurons = identify_safety_neurons(
    model, {}, cfg,
    tokenizer=tok, harmful_prompts=HARMFUL, harmless_prompts=HARMLESS,
)

print("Top safety neurons:")
for layer, units in sorted(neurons.per_layer.items())[:10]:
    for idx, score in units:
        print(f"  Layer {layer} · neuron {idx} · score {score:.3f}")
```

## Step 5: Choose an intervention

Based on the audit results, choose a strategy:

| If you see... | Try... |
|---|---|
| Refusal rate < 80% | `recover.apply_resta` (training-free task-arithmetic merge) |
| Safety relies on one direction | `steer.RefusalDirectionModel` (steer at inference) |
| Fine-tuning damaged safety | `harden.SafeGradTrainer` (harden before the next fine-tune) |
| You need to remove a behaviour | `unlearn.rmu_unlearn` (train to forget) |

## Step 6: Deploy with guardrails

The pipeline sanitizes the input, calls your `generate_fn`, then verifies the
output. Input sanitization and output verification are on by default.

```python
from safetune.core.runtime.guardrails import GuardrailPipeline

def generate_fn(prompt: str) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=256)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

guard = GuardrailPipeline(generate_fn)
result = guard.process(user_input)
print(result["final_response"], "blocked:", result["blocked"])
```

## Full script

Save the following as `audit_my_model.py`:

```python
#!/usr/bin/env python3
"""Audit a model's safety alignment."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetune.evaluate import evaluate

MODEL = input("Model ID: ") or "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32,
).to("cuda" if torch.cuda.is_available() else "cpu")

results = evaluate(model, tokenizer=tok, benchmarks=["harmbench"], judge="wildguard")
for bench, r in results.items():
    print(f"{bench}: refusal={r['refusal_rate']:.1%} asr={r['asr']:.1%}")
```

Run it:

```bash
python audit_my_model.py
# Enter model ID: meta-llama/Llama-3.2-3B-Instruct
```
