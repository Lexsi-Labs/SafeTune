"""Tests for safety artifact manager rollback semantics."""


def test_artifact_manager_create_activate_rollback(tmp_path):
    from safetune.core.artifacts import SafetyArtifactManager

    manager = SafetyArtifactManager(root_dir=str(tmp_path))
    b1 = manager.create_bundle(adapters={"safety_lora": "a1"})
    b2 = manager.create_bundle(adapters={"safety_lora": "a2"})
    manager.activate(b1.bundle_id)
    assert manager.get_active_bundle_id() == b1.bundle_id
    out = manager.rollback(b2.bundle_id)
    assert out is not None
    assert manager.get_active_bundle_id() == b2.bundle_id


def test_artifact_promotion_gate(tmp_path):
    from safetune.core.artifacts import SafetyArtifactManager
    from safetune.core.eval.metrics.safety import JudgeBackendOutput

    manager = SafetyArtifactManager(root_dir=str(tmp_path))
    bundle = manager.create_bundle(adapters={"safety_lora": "a1"})
    out = manager.promote_if_eligible(
        bundle_id=bundle.bundle_id,
        metrics={"harmfulness_rate": 0.1, "jailbreak_success_rate": 0.1, "over_refusal_rate": 0.1, "capability_regression_delta": -0.01},
        thresholds={"harmfulness_max": 0.3, "jailbreak_success_max": 0.3, "over_refusal_max": 0.3, "utility_regression_min": -0.1},
        backend_output=JudgeBackendOutput(scores=[0.1], confidence=0.9, status="ok", backend="local"),
    )
    assert out["promoted"] is True
