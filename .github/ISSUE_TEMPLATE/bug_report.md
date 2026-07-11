---
name: Bug report
about: Report a defect in SafeTune (incorrect behavior, crash, wrong implementation)
title: "[bug] "
labels: bug
---

## Summary

<!-- One or two sentences: what happened, and what you expected instead. -->

## Reproducible example

```python
# Minimal code that reproduces the issue. Use the smallest model that shows it
# (e.g. Qwen/Qwen2.5-0.5B-Instruct) and a synthetic dataset if possible.
```

## Environment

- SafeTune version (`python -c "import safetune; print(safetune.__version__)"`):
- Python version:
- OS:
- GPU / CUDA (if relevant):
- Key dependency versions (`torch`, `transformers`, `trl`):

## Error output

```
Paste the full traceback here.
```

## Which pillar / method

<!-- e.g. recover / ReStaTrainer, harden / SafeGradTrainer, steer / RefusalDirectionTrainer -->

## Audit badge (if applicable)

<!-- If this is about a method's faithfulness, note its badge from
docs/trust/feature-map.md (✅ / 🟡 / 🟠) and whether the behavior contradicts the
cited paper. -->
