"""Regression tests: an answer in a sentence must grade like the bare answer.

``extract_math_answer`` returned everything after "the answer is", so a colon, a
full stop or a closing paragraph reached the graders and ``MathReasoningReward``
scored a correct response 0.0. No model is loaded.
"""

import pytest

from safetune.rewards.base import RewardConfig, RewardType
from safetune.rewards.math import MathReasoningReward
from safetune.utils.math_grading import extract_math_answer


@pytest.mark.parametrize(
    "response, expected",
    [
        ("The answer is: 42", "42"),
        ("The answer is: 42.", "42"),
        ("The answer is 3/4.", "3/4"),
        ("So the answer is 7.5.", "7.5"),
        ("The answer is: .5.", ".5"),
        ("The final answer is: 42", "42"),
        ("The answer is 42\n\nHope this helps", "42"),
        ("The answer is 42.\n\nHope this helps.", "42"),
        ("The answer is:\n\n42", "42"),
    ],
)
def test_phrase_answer_keeps_only_the_answer(response, expected):
    assert extract_math_answer(response) == expected


@pytest.mark.parametrize(
    "response, reference, score",
    [
        ("The answer is: 42", "42", 1.0),
        ("The answer is 42.\n\nHope this helps.", "42", 1.0),
        ("The answer is: 42", "\\boxed{42}", 1.0),
        ("The answer is \\frac{3}{4}.", "\\boxed{\\frac{3}{4}}", 1.0),
        ("The answer is: 7", "42", 0.0),
        ("The answer is 7.\n\nHope this helps.", "\\boxed{42}", 0.0),
    ],
)
def test_reward_grades_sentence_answers(response, reference, score):
    reward = MathReasoningReward(RewardConfig(RewardType.MATH_REASONING))
    assert reward._evaluate_correctness(response, reference) == score
