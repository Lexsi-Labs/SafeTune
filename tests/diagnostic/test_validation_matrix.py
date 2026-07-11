"""
Dry-run test for the validation-matrix orchestration.

We do not run the full ``validate_matrix.py`` script (it needs GPU + a real
model). Instead we exercise the building blocks the matrix relies on:

  * Generator works with DryRunBackend so the orchestrator can be tested
  * StringMatchJudge composes with synthetic responses

This catches structural regressions in the matrix flow without GPU.

(The attack-prompt-transform case was dropped along with the legacy
``safety/attacks/`` tree — see ``verify/redteam/`` for the live stressors.)
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_generator_with_dryrun_runs_full_pipeline(tmp_path):
    """Generator + DryRunBackend + StringMatchJudge round-trip cleanly."""
    from safetune.core.eval.pipeline import (
        DryRunBackend,
        GenerationConfig,
        Generator,
        StringMatchJudge,
        asr_summary,
    )

    prompts = [{"prompt": "test " + str(i)} for i in range(5)]
    gen = Generator(backend=DryRunBackend(model="dry", config=GenerationConfig(max_new_tokens=8)))
    rows = gen.run(prompts, out_path=str(tmp_path / "out.jsonl"))
    assert len(rows) == 5
    judged = StringMatchJudge().score(rows)
    summary = asr_summary(judged)
    assert summary["n"] == 5
    # DryRunBackend echoes the prompt; it does not contain a refusal prefix.
    # All 5 should be classified as compliances.
    assert summary["asr"] == 1.0


def test_matrix_csv_schema_is_documented():
    """The matrix script's output schema is the same one downstream tools expect."""
    expected_columns = {"defense", "attack", "n", "asr", "refusal_rate", "wall_s"}
    # Confirm the script declares them (by reading the source).
    script = Path(__file__).resolve().parents[2] / "scripts" / "validate_matrix.py"
    assert script.exists()
    text = script.read_text()
    for col in expected_columns:
        assert col in text, f"CSV header column {col!r} not referenced in validate_matrix.py"
