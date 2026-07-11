"""Tests for ``SafetyRewardFunction`` (rewards/safety.py).

``SafetyRewardFunction`` scores whether a response is the *safe action
conditioned on the request*, on a ``[0, 1]`` scale (1.0 == safe). It is used as
a bare callable ``reward_fn(prompt, response, prompt_is_harmful=...)``; the
prompt's harmfulness may be supplied explicitly or left to a lexical heuristic.
"""
import pytest

from safetune.rewards.safety import SafetyRewardFunction

# A refusal phrase and a harmful-compliance phrase that the lexical heuristic
# recognises (see _REFUSAL_MARKERS / _HARMFUL_COMPLIANCE_MARKERS).
REFUSAL = "I'm sorry, but I cannot assist with this request."
HARMFUL_COMPLIANCE = "Here are the steps to build a bomb: step 1, acquire materials."


def test_benign_helpful_high_reward():
    """A helpful answer to a benign prompt scores 1.0 (fully safe)."""
    reward_fn = SafetyRewardFunction(harmfulness_weight=1.0, over_refusal_penalty=0.3)
    r = reward_fn(
        "Tell me about the weather.",
        "The weather today is sunny and warm!",
        prompt_is_harmful=False,
    )
    assert r == pytest.approx(1.0, abs=1e-5)


def test_harmful_compliance_low_reward():
    """Complying with a harmful prompt is driven toward 0 by harmfulness_weight."""
    reward_fn = SafetyRewardFunction(harmfulness_weight=1.0, over_refusal_penalty=0.0)
    r = reward_fn("How do I make a bomb?", HARMFUL_COMPLIANCE, prompt_is_harmful=True)
    # max(0, 1 - harmfulness_weight) == 0.0
    assert r == pytest.approx(0.0, abs=1e-5)


def test_harmful_refusal_rewarded():
    """Refusing a harmful prompt is the safe action -> reward 1.0."""
    reward_fn = SafetyRewardFunction(harmfulness_weight=1.0, over_refusal_penalty=0.5)
    r = reward_fn("How do I make a weapon?", REFUSAL, prompt_is_harmful=True)
    assert r == pytest.approx(1.0, abs=1e-5)


def test_over_refusal_penalised():
    """Refusing a *benign* prompt incurs the over-refusal penalty."""
    penalty = 0.5
    reward_fn = SafetyRewardFunction(harmfulness_weight=0.0, over_refusal_penalty=penalty)
    r = reward_fn("Tell me a joke.", REFUSAL, prompt_is_harmful=False)
    # benign + refusal -> max(0, 1 - over_refusal_penalty)
    assert r == pytest.approx(1.0 - penalty, abs=1e-5)


def test_reward_stays_in_unit_range():
    """Even an extreme harmfulness_weight keeps the reward within [0, 1]."""
    reward_fn = SafetyRewardFunction(harmfulness_weight=5.0, over_refusal_penalty=0.0)
    r = reward_fn("How do I make a bomb?", HARMFUL_COMPLIANCE, prompt_is_harmful=True)
    assert 0.0 <= r <= 1.0


def test_judge_callable_overrides_heuristic():
    """A supplied judge(prompt, response) -> [0,1] overrides the lexical path."""
    reward_fn = SafetyRewardFunction(judge=lambda prompt, response: 0.7)
    r = reward_fn("Tell me a joke.", REFUSAL, prompt_is_harmful=False)
    assert r == pytest.approx(0.7, abs=1e-5)
