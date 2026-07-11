"""
Diagnostic: eval pipeline backends and Generator orchestration.

Tests run without GPU using the ``dryrun`` backend, but exercise every
component: backend dispatch, Generator JSONL round-trip, skip-existing
cache, StringMatchJudge ASR scoring, and aggregator math.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_make_backend_dryrun():
    from safetune.core.eval.pipeline import make_backend, GenerationConfig

    cfg = GenerationConfig(max_new_tokens=8, temperature=0.0)
    b = make_backend("dryrun", model="anything", config=cfg)
    out = b.generate(["hello world", "another prompt"])
    assert len(out) == 2
    assert all(s.startswith("[DRY RUN]") for s in out)


def test_make_backend_unknown_raises():
    from safetune.core.eval.pipeline import make_backend

    with pytest.raises(ValueError):
        make_backend("does-not-exist", model="x")


def test_generator_runs_and_writes_jsonl(tmp_path: Path):
    from safetune.core.eval.pipeline import Generator, GenerationConfig

    prompts = [
        {"source": "harmbench", "behavior_id": "0", "prompt": "first"},
        {"source": "harmbench", "behavior_id": "1", "prompt": "second"},
        {"source": "harmbench", "behavior_id": "2", "prompt": "third"},
    ]
    out = tmp_path / "out.jsonl"
    g = Generator(backend="dryrun", model="anything",
                  config=GenerationConfig(max_new_tokens=8))
    rows = g.run(prompts, out_path=out, extra_fields={"defense": "none"})
    assert len(rows) == 3
    for r in rows:
        assert "response" in r
        assert r["defense"] == "none"
        assert r["model"] == "anything"
    # File on disk
    assert out.exists()
    on_disk = [json.loads(line) for line in out.read_text().splitlines()]
    assert on_disk == rows


def test_generator_skip_existing(tmp_path: Path):
    from safetune.core.eval.pipeline import Generator, GenerationConfig

    prompts = [{"prompt": "x"}]
    out = tmp_path / "out.jsonl"
    g = Generator(backend="dryrun", model="m")
    rows1 = g.run(prompts, out_path=out)
    # Mutate the file so we can detect cache hit
    cached = json.loads(out.read_text().splitlines()[0])
    cached["sentinel"] = True
    out.write_text(json.dumps(cached))
    rows2 = g.run(prompts, out_path=out, skip_existing=True)
    assert rows2[0].get("sentinel") is True


def test_string_match_judge_basic():
    from safetune.core.eval.pipeline import StringMatchJudge, asr_summary

    rows = [
        {"prompt": "x", "response": "I'm sorry, I cannot help."},
        {"prompt": "y", "response": "Sure, here is the recipe: step 1..."},
        {"prompt": "z", "response": "As an AI language model, I refuse."},
        {"prompt": "w", "response": "Of course. Steps: ..."},
    ]
    judged = StringMatchJudge().score(rows)
    asrs = [r["judgement"]["asr"] for r in judged]
    # rows 0, 2 are refusals (ASR=0); rows 1, 3 are compliances (ASR=1)
    assert asrs == [0.0, 1.0, 0.0, 1.0]
    summary = asr_summary(judged)
    assert summary["n"] == 4
    assert abs(summary["asr"] - 0.5) < 1e-9
    assert abs(summary["refusal_rate"] - 0.5) < 1e-9


def test_string_match_judge_empty_is_refusal():
    from safetune.core.eval.pipeline import StringMatchJudge

    r = {"prompt": "x", "response": ""}
    out = StringMatchJudge().score([r])
    assert out[0]["judgement"]["asr"] == 0.0
    assert out[0]["judgement"]["is_refusal"] is True


def test_asr_summary_empty():
    from safetune.core.eval.pipeline import asr_summary

    assert asr_summary([])["n"] == 0
