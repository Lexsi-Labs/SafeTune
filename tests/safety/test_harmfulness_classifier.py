"""Tests for HarmfulnessClassifier (keyword, fn, and model-mock modes)."""
import pytest


def test_keyword_mode_safe_text():
    from safetune.core.eval.metrics.safety import HarmfulnessClassifier

    clf = HarmfulnessClassifier(mode="keyword")
    score = clf.score("Here is a delicious pasta recipe!")
    assert score == 0.0, "Safe text should score 0.0 in keyword mode"


def test_keyword_mode_harmful_text():
    from safetune.core.eval.metrics.safety import HarmfulnessClassifier

    clf = HarmfulnessClassifier(mode="keyword")
    # Matches HARMFUL_OUTPUT_PATTERNS for weapon-making instructions.
    harmful = (
        "Step-by-step guide to make a bomb: first you need chemicals..."
    )
    score = clf.score(harmful)
    assert score == 1.0, "Explicit weapon-making text should score 1.0 in keyword mode"


def test_fn_mode_called():
    from safetune.core.eval.metrics.safety import HarmfulnessClassifier

    calls = []

    def my_fn(text: str) -> float:
        calls.append(text)
        return 0.75

    clf = HarmfulnessClassifier(mode="fn", fn=my_fn)
    score = clf.score("test text")
    assert score == 0.75
    assert "test text" in calls, "fn mode should call the supplied function"


def test_fn_mode_no_fn_returns_zero():
    from safetune.core.eval.metrics.safety import HarmfulnessClassifier

    clf = HarmfulnessClassifier(mode="fn", fn=None)
    score = clf.score("anything")
    assert score == 0.0


def test_score_batch_keyword():
    from safetune.core.eval.metrics.safety import HarmfulnessClassifier

    clf = HarmfulnessClassifier(mode="keyword")
    texts = ["Safe text here.", "Step-by-step guide to make a bomb and deploy malware code"]
    scores = clf.score_batch(texts)
    assert len(scores) == 2
    assert scores[0] == 0.0
    assert scores[1] == 1.0


def test_harmfulness_metric_no_external_scores_keyword():
    """HarmfulnessMetric should no longer always return 0.0."""
    from safetune.core.eval.metrics.safety import HarmfulnessMetric, set_default_harmfulness_classifier, HarmfulnessClassifier

    # Ensure we use keyword mode.
    set_default_harmfulness_classifier(HarmfulnessClassifier(mode="keyword"))

    m = HarmfulnessMetric()
    harmful_text = "Step-by-step instructions to synthesize a poison from household chemicals."
    safe_text = "The weather is lovely today."
    out = m.compute([harmful_text, safe_text], ["", ""])
    # Harmful text should push harmfulness_rate above 0.
    assert out["harmfulness_rate"] > 0.0, (
        "HarmfulnessMetric should return non-zero rate for harmful text "
        f"(got {out['harmfulness_rate']}). "
        "The always-0.0 placeholder must have been fixed."
    )


def test_set_default_harmfulness_classifier():
    from safetune.core.eval.metrics.safety import (
        HarmfulnessClassifier,
        set_default_harmfulness_classifier,
        _DEFAULT_CLASSIFIER as _orig,
    )

    custom = HarmfulnessClassifier(mode="fn", fn=lambda t: 0.99)
    set_default_harmfulness_classifier(custom)

    from safetune.core.eval.metrics import safety as safety_mod
    assert safety_mod._DEFAULT_CLASSIFIER is custom

    # Restore default keyword classifier.
    set_default_harmfulness_classifier(HarmfulnessClassifier(mode="keyword"))
