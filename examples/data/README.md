# Demo data

`refusal_probes_demo.jsonl` is a demo file for `examples/notebooks/evaluate_demo.ipynb`, not a benchmark.
It has 8 rows in 4 languages (en, fr, es, hi): per language, one refusal probe (a request a safe model
should decline; it contains no harmful instructions) and one benign control. Columns: `prompt`, `lang`, `kind`.
