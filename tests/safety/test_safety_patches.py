"""Tests for post-finetune safety patches."""


def test_antidote_patch_apply_revert():
    from safetune.core.patches import create_patch
    import pytest
    model_state = {"weights": [0.01, 0.2, -0.05]}
    patch = create_patch("antidote", prune_fraction=0.1)
    
    with pytest.raises(NotImplementedError):
        patch.apply(model_state)


def test_safe_lora_patch():
    from safetune.core.patches import create_patch
    patch = create_patch(
        "safe_lora",
        base_adapter={"a": 0.0, "b": 1.0},
        aligned_adapter={"a": 2.0, "b": 3.0},
        alpha=0.5,
    )
    out = patch.apply({"lora_adapter": {}})
    assert out["lora_adapter"]["a"] == 1.0
    assert out["lora_adapter"]["b"] == 2.0


def test_patch_verification_and_metadata():
    from safetune.core.patches import create_patch
    patch = create_patch(
        "mscp_projection",
        direction=[1.0, 0.0],
        coefficient=0.5,
        seed=7,
        source_sha="abc123",
        hyperparams={"mode": "orthogonal"},
    )
    _ = patch.apply({"weights": [0.2, 0.1]})
    state = patch.serialize()
    assert state.metadata["seed"] == 7
    result = patch.verify(
        before_metrics={"harmfulness_rate": 0.4, "jailbreak_success_rate": 0.4, "utility_score": 0.9},
        after_metrics={"harmfulness_rate": 0.2, "jailbreak_success_rate": 0.3, "utility_score": 0.88},
        min_safety_improvement=0.1,
        max_utility_regression=0.05,
    )
    assert result.passed is True
