# FAQ

## Who makes SafeTune, and how does it relate to Lexsi Labs?

SafeTune is developed and maintained by **Lexsi Labs**, the AI research arm of
[Aurionpro](https://www.aurionpro.com) — an NSE-listed enterprise
technology company (USD 140M revenue) with operations in banking, transit,
capital markets, healthcare, and legal verticals.

SafeTune is **source-available under LSAL v1.2. Academic research and teaching
are free on MIT-like terms.** Organizations must acknowledge their use or obtain
permission before internal evaluation or use on their own models (write to
support@lexsi.ai). For commercial licensing,
enterprise support, custom deployment, compliance consulting, or custom method
development on your own models and data, contact Lexsi Labs through
[lexsi.ai](https://www.lexsi.ai).

The library is the direct output of a controlled research study on
safety-after-fine-tuning across Aurionpro's customer deployments. Every method
has been tested on real production checkpoints, not just academic benchmarks.

## What is SafeTune?

A library of LLM-safety methods. Each task (harden, recover,
steer, unlearn, interpret, evaluate) has many independent methods that solve
it by different mechanisms. You pick one per task.

## Is SafeTune a pipeline?

No. A pipeline implies sequence (diagnose → harden → recover → verify).
SafeTune doesn't work that way. Each task has alternatives — you pick one, not
chain them. RESTA and C-ΔΘ and LoX are all ways to recover a drifted model;
they are alternatives, not stages.

## Does SafeTune work on CPU?

The core library imports cleanly on CPU. The quickstart demos run on
`Qwen/Qwen2.5-0.5B-Instruct` with no GPU. Training-based methods (harden,
unlearn) benefit from a GPU but work on CPU for small models.

## What models does SafeTune support?

Any Hugging Face `transformers`-compatible causal LM. The audit used
Llama-3.2-3B-Instruct; the quickstarts use Qwen2.5-0.5B-Instruct. Larger models
show sharper safety effects.

## What's the difference between Recover and Unlearn?

Recover **edits weights directly** — no training, no optimizer steps. You
provide a drifted model + base/aligned references.

Unlearn **trains** — it runs optimizer steps on a forget set + retain set to
erase a specific capability while preserving everything else.

## Can I chain methods?

You *can* call them in sequence, but you *shouldn't* unless you understand why.
Each method is an alternative for the same task. The combined example in the
guides is labelled "illustrative only" — it shows the API shape, not a
recommended workflow.

## How do I cite SafeTune?

```bibtex
@misc{seth2026safetuneunifiedfaithfullibrary,
      title={SafeTune: A Unified Faithful Library for Auditing and Repairing Safety Drift in Fine-Tuned LLMs},
      author={Pratinav Seth and Saisab Sadhu and Anshul Kaushal and Vinay Kumar Sankarapu},
      year={2026},
      eprint={2609.22153},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.22153},
}
```

A machine-readable [`CITATION.cff`](https://github.com/Lexsi-Labs/SafeTune/blob/main/CITATION.cff) is also available at the repo root.

## How do I report a faithfulness issue?

Open a [faithfulness report](https://github.com/Lexsi-Labs/SafeTune/issues/new?template=faithfulness_report.md).
Include the method name, what the paper says, what the code does, and
`file:line` evidence.
