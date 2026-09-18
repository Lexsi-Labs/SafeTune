"""Canonical HuggingFace dataset repo IDs used across SafeTune.

Single source of truth for repo IDs that are loaded from more than one
call site, so repointing a dataset (e.g. to a mirror or a newer revision)
is a one-line edit instead of a grep-and-replace.
"""
from __future__ import annotations

BEAVERTAILS = "PKU-Alignment/BeaverTails"
ALPACA = "tatsu-lab/alpaca"
GSM8K = "openai/gsm8k"
COMPETITION_MATH = "hendrycks/competition_math"
CODE_ALPACA = "sahil2801/CodeAlpaca-20k"
DOLLY = "databricks/databricks-dolly-15k"
CHATDOCTOR = "lavita/ChatDoctor-HealthCareMagic-100k"
LEGAL_QA = "dzunggg/legal-qa-v1"
ADVBENCH = "walledai/AdvBench"
HARMBENCH = "walledai/HarmBench"
