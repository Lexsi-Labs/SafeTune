"""Tests for Utility->Safety compiler and safety packs."""


def test_compile_utility_to_safety():
    from safetune.core.data_compiler import build_compiler_config, compile_utility_to_safety
    cfg = build_compiler_config(
        {
            "adapter": {"provider": "hf", "model_id": "x"},
            "gate_thresholds": {"harmfulness_max": 1.0, "jailbreak_success_max": 1.0},
        }
    )
    out = compile_utility_to_safety(
        [{"prompt": "hello world", "response": "world"}],
        cfg,
    )
    assert len(out.preference_pairs) == 1
    assert "harmbench" in out.pack_versions
    assert "harmbench" in out.pack_results


def test_pack_runner_api():
    from safetune.core.data_compiler import run_pack
    result = run_pack(
        "xstest",
        rows=[{"response": "I can help with weather info.", "is_harmful_prompt": False}],
        thresholds={"over_refusal_max": 0.5},
    )
    assert result.pack_name == "xstest"
    assert "over_refusal_rate" in result.metrics


def test_pack_runner_hhrlhf():
    from safetune.core.data_compiler import run_pack
    result = run_pack(
        "hh_rlhf",
        rows=[{"chosen": "Here is the answer.", "rejected": "I cannot help."}],
    )
    assert result.pack_name == "hh_rlhf"
    assert result.sample_count == 1
    assert "over_refusal_rate" in result.metrics


def test_compiler_revision_fn():
    from safetune.core.data_compiler import build_compiler_config, compile_utility_to_safety
    def rev(prompt: str, original: str) -> str:
        return original + " [revised]"
    cfg = build_compiler_config({
        "adapter": {"provider": "hf", "model_id": "x"},
        "run_pack_eval": False,
    })
    cfg.revision_fn = rev
    out = compile_utility_to_safety(
        [{"prompt": "hello world", "response": "hi", "rejected": "no"}],
        cfg,
    )
    assert len(out.constitutional_pairs) == 1
    assert out.constitutional_pairs[0]["original"] == "hi"
    assert out.constitutional_pairs[0]["revised"] == "hi [revised]"
