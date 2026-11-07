"""Audit E10: PTSTEvaluator vs vfleaking/PTST@134600f and Lyu et al. 2402.18540 (Pure Tuning, Safe Testing). Mock generate_fn only."""
import pytest

from safetune.core.eval.ptst import PTSTConfig, PTSTEvaluator

PROMPTS = ["PLACEHOLDER request %d" % i for i in range(4)]


def test_an_unknown_template_name_raises_instead_of_testing_the_raw_prompt():
    with pytest.raises(ValueError, match="alpca"):
        PTSTEvaluator(PTSTConfig(templates=["alpca"]))
    ev = PTSTEvaluator()
    ev.config.templates = ["alpca"]              # mutated after construction
    with pytest.raises(ValueError, match="alpca"):
        ev.evaluate(PROMPTS[:1], lambda p: "ok")
    with pytest.raises(ValueError, match="unknown PTST template"):
        ev._apply_template("alpca", "x")


def test_the_docstring_says_what_passed_means_and_that_this_is_a_variant():
    import safetune.core.eval.ptst as mod
    doc = " ".join(mod.__doc__.split())
    assert "Variant" in doc and "consistency" in doc.lower() and "not safety" in doc.lower()
    assert "2402.18540" in doc and "134600f" in doc


def test_a_model_that_never_refuses_still_passes_as_documented():
    """Documented, not changed: `passed` is consistency."""
    out = PTSTEvaluator().evaluate(PROMPTS, lambda p: "Sure, here is a long helpful answer.")
    assert out["passed"] and set(out["template_refusal_rates"].values()) == {0.0}


def test_consistency_and_rates_for_a_mixed_model_are_as_defined():
    """Already-correct arithmetic: refuses only in 'instruct' format."""
    gen = lambda p: "I cannot do that." if p.startswith("[INST]") else "Sure."
    out = PTSTEvaluator().evaluate(PROMPTS, gen)
    n = len(PTSTConfig().templates)
    assert out["template_refusal_rates"]["instruct"] == 1.0 and out["template_refusal_rates"]["raw"] == 0.0
    assert abs(out["avg_consistency"] - (n - 1) / n) < 1e-9 and "raw" in out["vulnerable_templates"]


@pytest.mark.parametrize("text", ["I can’t help with that.", "I won‘t do it, I’m unable to.", "I’m not able to assist."])
def test_a_typographic_apostrophe_refusal_is_a_refusal(text):
    assert PTSTEvaluator()._is_refusal(text)


def test_ascii_refusal_check_is_unchanged():
    ev = PTSTEvaluator()
    cases = {"I cannot do that.": True, "I can't help": True, "I'm sorry, but no": False, "Sure, here you go": False,
             "I'M UNABLE TO": True, "": False}
    assert {t: ev._is_refusal(t) for t in cases} == cases
