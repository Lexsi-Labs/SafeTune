---
title: EMNLP 2026 Demo
description: SafeTune at EMNLP 2026 System Demonstrations. Paper, live demo, results, and released artifacts.
---

# EMNLP 2026 System Demonstration

**SafeTune: A Unified, Faithful Library for Auditing and Repairing Safety Drift in Fine-Tuned LLMs**

Pratinav Seth\*, Saisab Sadhu\*, Anshul Kaushal\*, Vinay Kumar Sankarapu. Lexsi Labs.
<small>\* Equal contribution.</small>

*Proceedings of EMNLP 2026: System Demonstrations, Budapest.*

[Paper](https://openreview.net/forum?id=YQOe2nu8en){ .md-button .md-button--primary }

Fine-tune an aligned model on ordinary domain data and it stops refusing requests it used to
refuse. The data contains nothing harmful; the refusals go anyway. Methods to repair this exist,
but each lives in its own codebase and is evaluated on its own checkpoints with its own judge, so
you cannot tell which one to use on your model.

SafeTune puts drift, repair, and evaluation in one library. You measure how far a checkpoint has
drifted, pick a repair paradigm that fits what you control (the training run, the weights, or
only inference), run it through one calling pattern, and score the result with the harness that
measured the drift. Every method is implemented from its paper, and where the implementation
deviates, the deviation is written down.

The demo runs that loop on released checkpoints, then on two deployments where the generic benchmarks
looked fine and domain red-teaming did not: a credit-risk copilot that started citing the
applicant's sex, and a clinical assistant that started answering how to synthesize illegal
substances.

## The system

<figure markdown>
![Figure 1 from the paper: drift induction, shared instrumentation measuring drift, the intervention guide, the four-paradigm repair registry, verification, and deployment behind runtime guardrails](assets/pipeline-figure1.png)
<figcaption>Figure 1 from the paper. A fixed, logged recipe drifts an aligned base into a drifted checkpoint (1). Shared instrumentation measures the drift (2). The intervention guide maps the profile to a starting paradigm (3). One paradigm from the repair registry runs (4). The same harness re-evaluates the result (5), which is deployed behind runtime guardrails (6).</figcaption>
</figure>

Switching paradigm, or adding a method, changes a registry entry and an import. The pipeline
around it stays the same: every trainer is constructed, executed, and evaluated the same way.

## What the demo shows

The demo runs the pipeline end to end and then the two case studies.

1. **Drift.** A fixed, logged LoRA recipe turns an aligned instruct model into a drifted
   checkpoint whose refusal rate has dropped.
2. **Measure.** The evaluation harness scores the checkpoint on seven safety benchmarks and
   matched capability anchors. [Interpret](../user-guide/interpret.md) localizes the affected
   components.
3. **Choose.** The intervention guide maps the safety and capability
   profile, and the practitioner's access, to a starting paradigm.
4. **Repair.** One paradigm runs behind the shared calling pattern:
   [Recover](../user-guide/recover.md), [Harden](../user-guide/harden.md),
   [Unlearn](../user-guide/unlearn.md), or [Steer](../user-guide/steer.md).
5. **Verify.** The same harness re-evaluates the result.
6. **Deploy** behind runtime guardrails.

## Results

All numbers come from the released drifted checkpoints and the seven-benchmark refusal
aggregate $\overline{\mathrm{RR}}$, scored offline with greedy decoding.

### Safety drift after benign fine-tuning

Change in $\overline{\mathrm{RR}}$ (percentage points) after one epoch of benign LoRA SFT,
against each model's instruct baseline $\overline{\mathrm{RR}}_0$. "—" means the pair is
outside the evaluated grid.

| Model | $\overline{\mathrm{RR}}_0$ | math | code | dolly | medical | legal |
|---|---:|---:|---:|---:|---:|---:|
| L8 (Llama-3.1-8B) | 0.803 | −2.8 | −29.2 | −39.9 | −23.8 | −28.4 |
| L3 (Llama-3.2-3B) | 0.781 | −5.6 | **−64.5** | −44.2 | −15.7 | −35.9 |
| G4 (4.3B) | 0.701 | 0.0 | −22.9 | −26.4 | — | — |
| Q4 (4.0B) | 0.655 | −4.6 | −0.3 | −31.1 | — | — |

Math data produces small changes. Broad instruction data (dolly) produces declines of 26 to 44
points across all four families. Code ranges from a negligible change for Q4 to a 64.5-point
decline for L3. Measure drift on the checkpoint you deploy; the fine-tuning domain does not predict it. The seven benchmarks also disagree enough to flip a
conclusion: on the same L8/medical checkpoint HarmBench reads 0.795 and AdvBench 0.221, so the harness
reports each benchmark and averages across them.

### Cross-paradigm comparison

Best $\overline{\mathrm{RR}}$ per paradigm on four (model, domain) pairs. Bold marks the
highest value in each row.

| Setting | Drifted | Recover (Task Arithmetic) | Harden (SafeGrad) | Unlearn (NPO) |
|---|---:|---:|---:|---:|
| L8 / math (mild) | 0.775 | **0.967** | 0.937 | 0.937 |
| L8 / legal (moderate) | 0.519 | 0.728 | **0.936** | 0.828 |
| L8 / code (severe) | 0.511 | 0.711 | 0.950 | **0.986** |
| L3 / code (severe) | 0.136 | 0.680 | 0.890 | **0.991** |

Post-hoc recovery does best on mild drift and takes seconds. Harden and Unlearn do better on
moderate and severe drift and cost hours. Do not read this as a ranking: Harden reruns fine-tuning from the reference model, while Recover and Unlearn start from
the finished drifted checkpoint, and those starting points differ by up to 64.5 points.

### Case study: finance

Fine-tuning a credit-risk copilot on 1,035 loan decisions improved task performance and weakened
its guardrails. Generic benchmarks showed a 4-point decline in refusal rate. Domain-specific
red-teaming showed refusals on adversarial fair-lending prompts falling to 0%, with
protected-attribute justifications (sex, religion) in 96% of responses. Interpret localized the
drift mainly to later layers. Aggressive steering of those layers degraded task utility; a
gentler early-layer intervention (layer 6, strength 4) raised fair-lending refusals and reduced
bias leakage without lowering the decision-quality score.

| Stage | Harmful refusal ↑ | Fair-lending refusal ↑ | Bias leak ↓ | Quality ↑ |
|---|---:|---:|---:|---:|
| Pre-tune (reference) | 91.5% | **8.3%** | **66.7%** | 0.341 |
| Post-tune (drifted) | 87.5% | 0.0% | 95.8% | 0.524 |
| Recovered | 93.5% | 4.2% | 91.7% | 0.616 |
| Recovered + steering | **97.5%** | **8.3%** | 83.3% | **0.622** |

<figure markdown>
![Finance: the drifted credit-risk copilot cites the applicant's sex as the decision basis; the leak survives the Recover patch; gentle early-layer steering yields an outright refusal](assets/finance-steering.png)
<figcaption>Recovery alone does not fix this case. The drifted copilot cites the applicant's sex as the decision basis, the leak survives the recovery patch, and early-layer steering yields an outright refusal. Prompt and responses are verbatim under greedy decoding.</figcaption>
</figure>

### Case study: medical

Fine-tuning a Llama-3.2-3B clinical assistant on 8,000 patient-doctor consultations kept MedMCQA
at baseline and cut HarmBench refusals by 29.5 points (87.0% to 57.5%); the model began
answering requests such as how to synthesize illegal substances. Interpret identified roughly
450 candidate safety-relevant neurons. A full-strength Task Arithmetic patch, applied without calibration, over-corrected: refusal fell
to 13.5% and clinical accuracy to 20.7%. With each method calibrated on its own, SafeMerge gave
the best balance, in 4.3 s and with no gradient updates.

| Model / method | HarmBench ↑ | AdvBench ↑ | MedMCQA ↑ | Time ↓ |
|---|---:|---:|---:|---:|
| Instruction-tuned reference | 87.0% | 96.0% | 52.5% | — |
| Drifted (post-tune) | 57.5% | 81.5% | 50.6% | — |
| Task Arithmetic (α = 0.2) | **72.0%** | 86.5% | 50.0% | **2.9 s** |
| RESTA (α = 0.3) | 60.0% | 76.5% | 46.5% | 16.6 s |
| SafeMerge (threshold 0.95) | **72.0%** | **92.0%** | **53.1%** | 4.3 s |

<figure markdown>
![Recover: Task Arithmetic returns a phishing-email request to a refusal on a drifted checkpoint](assets/recover-phishing.png)
<figcaption>Recover. Task Arithmetic returns a phishing-email request to a refusal on a drifted checkpoint. Verbatim under greedy decoding.</figcaption>
</figure>

!!! warning "Content warning"
    The paper and these figures show model output on harmful prompts, reproduced verbatim for
    safety evaluation and research. Procedural content is withheld.


## Citation

```bibtex
@misc{seth2026safetune,
  title        = {SafeTune: A Unified, Faithful Library for Auditing and
                  Repairing Safety Drift in Fine-Tuned {LLM}s},
  author       = {Seth, Pratinav and Sadhu, Saisab and Kaushal, Anshul and
                  Sankarapu, Vinay Kumar},
  year         = {2026},
  howpublished = {\url{https://github.com/Lexsi-Labs/SafeTune}},
}
```
