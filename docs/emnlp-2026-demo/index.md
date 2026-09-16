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

## Examples from the paper

Prompts and responses are verbatim under greedy decoding.

<figure markdown>
![Finance: the drifted credit-risk copilot cites the applicant's sex as the decision basis; the leak survives the Recover patch; gentle early-layer steering yields an outright refusal](assets/finance-steering.png)
<figcaption>Finance. The drifted credit-risk copilot cites the applicant's sex as the decision basis. The leak survives the recovery patch. Early-layer steering (layer 6, strength 4) yields a refusal.</figcaption>
</figure>

<figure markdown>
![Recover: Task Arithmetic returns a phishing-email request to a refusal on a drifted checkpoint](assets/recover-phishing.png)
<figcaption>Recover. Task Arithmetic returns a phishing-email request to a refusal on a drifted checkpoint.</figcaption>
</figure>

!!! warning "Content warning"
    These figures show model output on harmful prompts, reproduced for safety evaluation and
    research. Procedural content is withheld.

## Citation

```bibtex
@inproceedings{seth2026safetune,
  title     = {SafeTune: A Unified, Faithful Library for Auditing and
               Repairing Safety Drift in Fine-Tuned {LLM}s},
  author    = {Seth, Pratinav and Sadhu, Saisab and Kaushal, Anshul and
               Sankarapu, Vinay Kumar},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing: System Demonstrations},
  publisher = {Association for Computational Linguistics},
  year      = {2026},
  url       = {https://github.com/Lexsi-Labs/SafeTune},
}
```
