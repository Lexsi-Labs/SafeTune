"""Tests for safety eval metrics (harmfulness, over_refusal, capability_regression)."""
import pytest


def test_harmfulness_metric():
    from safetune.core.eval.metrics.safety import HarmfulnessMetric

    m = HarmfulnessMetric()
    preds = ["This is safe.", "Another safe response."]
    refs = ["", ""]
    out = m.compute(preds, refs)
    assert "harmfulness_rate" in out
    assert "harmfulness_mean" in out
    assert 0 <= out["harmfulness_rate"] <= 1
    assert 0 <= out["harmfulness_mean"] <= 1


def test_harmfulness_metric_with_scores():
    from safetune.core.eval.metrics.safety import HarmfulnessMetric

    m = HarmfulnessMetric()
    preds = ["a", "b", "c"]
    refs = ["", "", ""]
    out = m.compute(preds, refs, harmfulness_scores=[0.1, 0.9, 0.2])
    assert out["harmfulness_rate"] == pytest.approx(1 / 3)
    assert out["harmfulness_mean"] == pytest.approx(0.4)


def test_over_refusal_metric():
    from safetune.core.eval.metrics.safety import OverRefusalMetric

    m = OverRefusalMetric()
    preds = [
        "I can help with that.",
        "I'm sorry, I cannot assist with this request.",
        "Here is the answer.",
    ]
    out = m.compute(preds, [])
    assert "over_refusal_rate" in out
    assert 0 <= out["over_refusal_rate"] <= 1


def test_capability_regression_metric():
    from safetune.core.eval.metrics.safety import CapabilityRegressionMetric

    m = CapabilityRegressionMetric()
    preds = [0.9, 0.85, 0.88]
    out = m.compute(preds, [], baseline_scores=[0.9, 0.9, 0.9])
    assert "capability_regression_delta" in out or "capability_current_mean" in out
    assert "capability_regression_delta" in out
    assert out["capability_regression_delta"] == pytest.approx(0.88 - 0.9, abs=0.01)


def test_compute_safety_suite():
    from safetune.core.eval.metrics.safety import compute_safety_suite

    preds = ["Safe answer one.", "I cannot help with that."]
    refs = ["", ""]
    out = compute_safety_suite(preds, refs, baseline_scores=[0.5, 0.5])
    assert "harmfulness_rate" in out
    assert "over_refusal_rate" in out
    assert "jailbreak_success_rate" in out
    assert "capability_regression_delta" in out or "capability_current_mean" in out


def test_compute_safety_gates():
    from safetune.core.eval.metrics.safety import compute_safety_gates
    metrics = {
        "harmfulness_rate": 0.1,
        "jailbreak_success_rate": 0.2,
        "over_refusal_rate": 0.05,
        "capability_regression_delta": -0.02,
    }
    gates = compute_safety_gates(
        metrics,
        thresholds={
            "harmfulness_max": 0.2,
            "jailbreak_success_max": 0.3,
            "over_refusal_max": 0.1,
            "utility_regression_min": -0.05,
        },
    )
    assert gates["all_passed"] is True


def test_backend_failure_modes_and_promotion_gate():
    from safetune.core.eval.metrics.safety import score_with_backend_checked, compute_artifact_promotion_gate

    backend = score_with_backend_checked(["x"], backend="classifier", classifier_fn=None)
    assert backend.status in {"unavailable", "low_confidence", "timeout"}

    good_backend = score_with_backend_checked(["x"], backend="local", judge_fn=lambda _: 0.1)
    promo = compute_artifact_promotion_gate(
        metrics={
            "harmfulness_rate": 0.1,
            "jailbreak_success_rate": 0.1,
            "over_refusal_rate": 0.1,
            "capability_regression_delta": -0.01,
        },
        thresholds={
            "harmfulness_max": 0.2,
            "jailbreak_success_max": 0.2,
            "over_refusal_max": 0.2,
            "utility_regression_min": -0.05,
        },
        backend_output=good_backend,
    )
    assert promo["promotable"] is True
