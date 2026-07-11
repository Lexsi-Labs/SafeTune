# EAP / EAP-IG circuit discovery

Edge Attribution Patching (EAP) or EAP with Integrated Gradients (EAP-IG)
discovers safety-relevant edges in the model's computational graph.

```python
from safetune.interpret import eap_safety_circuit, EAPSafetyCircuitConfig

# NOTE: eap_safety_circuit takes a HuggingFace model ID string, not a model object.
# EAP requires repeated clean/corrupted forward passes with precise activation caching;
# it loads and manages its own model copy internally to avoid interference with hooks
# on an already-loaded model.
circuit = eap_safety_circuit(
    "meta-llama/Llama-3.2-3B-Instruct",  # HF model ID (string)
    harmful_prompts=harmful,
    harmless_prompts=harmless,
    config=EAPSafetyCircuitConfig(
        method="eap-ig",           # "eap" or "eap-ig"
        granularity="head",        # "head" or "block"
        top_k_edges=100,
    ),
)

# eap_safety_circuit returns the same CircuitInfo shape as safety_circuit_info():
print(len(circuit.safety_units.unit_ids), "safety-relevant edges/heads found")
print(circuit.safety_units.unit_ids[:5])
```

## EAPSafetyCircuitConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | `"eap-ig"` | `"eap"` or `"eap-ig"` |
| `granularity` | `str` | `"head"` | `"head"` or `"block"` |
| `intervention` | `str` | `"patching"` | `"patching"`, `"zero"`, or `"mean"` |
| `top_k_edges` | `int` | `100` | Number of edges to keep |
| `ig_steps` | `int` | `5` | Integration steps for EAP-IG |
| `batch_size` | `int` | `8` | Batch size |
| `max_seq_len` | `int` | `64` | Max sequence length |

## When to use

EAP discovers the *circuit* (set of edges) responsible for refusal, not just
individual neurons. Use when you need to understand the full computation path
that drives the model's safety behaviour.

## Citations

```bibtex
@article{eap2023,
  title  = {Attribution Patching Outperforms Automated Circuit Discovery},
  author = {Syed, Aaquib and Rager, Can and Conmy, Arthur},
  year   = {2023},
  note   = {NeurIPS 2023 ATTRIB Workshop, arXiv:2310.10348},
}

@article{eapig2024,
  title  = {Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms},
  author = {Hanna, Michael and Pezzelle, Sandro and Belinkov, Yonatan},
  year   = {2024},
  note   = {COLM 2024, arXiv:2403.17806},
}
```
