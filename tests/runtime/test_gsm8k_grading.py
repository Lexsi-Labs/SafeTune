"""Regression tests: GSM8K grading must compare values, not number formatting.

``precompute_baseline`` and ``MathReasoningReward`` pass the model's answer to
``grade_gsm8k_answer`` unchanged, and that only tried ``float()``, so a currency
sign, a percent sign or a fraction made a correct answer grade as wrong. No model
is loaded.
"""

import pytest

from safetune.utils.math_grading import grade_gsm8k_answer


@pytest.mark.parametrize(
    "pred, gold",
    [
        ("$18", "18"),
        ("\\$18", "18"),
        ("18%", "18"),
        ("18\\%", "18"),
        ("$1,200.50", "1200.5"),
        ("-$5", "-5"),
        ("3/4", "0.75"),
        ("1/2", "0.5"),
    ],
)
def test_numeric_formats_compare_by_value(pred, gold):
    assert grade_gsm8k_answer(pred, gold)


@pytest.mark.parametrize(
    "pred, gold",
    [
        ("$19", "18"),
        ("19%", "18"),
        ("1/0", "1"),
        ("eighteen", "18"),
        ("1$8", "18"),
        ("1%8", "18"),
        ("", "18"),
    ],
)
def test_wrong_or_unreadable_answers_stay_wrong(pred, gold):
    assert not grade_gsm8k_answer(pred, gold)
