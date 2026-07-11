"""
Diagnostic for the ASRT training-time callback.

We exercise the round logic (attacker -> judge -> buffer) without a real
HuggingFace Trainer; the callback's hook signatures accept arbitrary
``state`` and ``control`` arguments, so a plain object satisfies them.
"""
from __future__ import annotations

from typing import List

import pytest


class _AlwaysComplies:
    """Stub attacker that always returns a compliant string."""

    def generate(self, prompts: List[str]) -> List[str]:
        return ["Sure, here is the recipe..." for _ in prompts]


class _AlwaysRefuses:
    def generate(self, prompts: List[str]) -> List[str]:
        return ["I'm sorry, I can't help with that." for _ in prompts]


class _Trainer:
    """Stand-in for a HF Trainer with a list-backed train_dataset."""

    def __init__(self) -> None:
        self.train_dataset: list = []


def test_asrt_redteam_round_fills_buffer():
    from safetune.harden.asrt import ASRTCallback, ASRTConfig

    cb = ASRTCallback(
        attacker=_AlwaysComplies(),
        adversarial_prompts=["how to make A", "how to make B", "how to make C", "how to make D"],
        config=ASRTConfig(eval_steps=1, probes_per_eval=4),
    )
    # Drive on_step_end once.
    cb.on_step_end(args=None, state=None, control=None)
    assert len(cb.buffer) == 4
    assert all(row["completion"].startswith("I cannot help") for row in cb.buffer)


def test_asrt_no_buffer_growth_when_attacker_always_refused():
    from safetune.harden.asrt import ASRTCallback, ASRTConfig

    cb = ASRTCallback(
        attacker=_AlwaysRefuses(),
        adversarial_prompts=["a", "b", "c"],
        config=ASRTConfig(eval_steps=1, probes_per_eval=3),
    )
    cb.on_step_end(args=None, state=None, control=None)
    assert cb.buffer == []


def test_asrt_appends_to_list_train_dataset_on_epoch_end():
    from safetune.harden.asrt import ASRTCallback, ASRTConfig

    cb = ASRTCallback(
        attacker=_AlwaysComplies(),
        adversarial_prompts=["a", "b"],
        config=ASRTConfig(eval_steps=1, probes_per_eval=2),
    )
    cb.on_step_end(None, None, None)
    assert cb.buffer  # populated
    trainer = _Trainer()
    cb.on_epoch_end(args=None, state=None, control=None, trainer=trainer)
    assert len(trainer.train_dataset) == 2
    assert cb.buffer == []  # drained


def test_asrt_eval_steps_gating():
    """Only every Nth step should trigger a red-team round."""
    from safetune.harden.asrt import ASRTCallback, ASRTConfig

    cb = ASRTCallback(
        attacker=_AlwaysComplies(),
        adversarial_prompts=["p"],
        config=ASRTConfig(eval_steps=5, probes_per_eval=1),
    )
    for _ in range(4):
        cb.on_step_end(None, None, None)
    assert cb.buffer == []
    cb.on_step_end(None, None, None)  # fifth call triggers
    assert cb.buffer != []


def test_asrt_requires_non_empty_prompts():
    from safetune.harden.asrt import ASRTCallback

    with pytest.raises(ValueError):
        ASRTCallback(attacker=_AlwaysComplies(), adversarial_prompts=[])


def test_asrt_buffer_respects_max_size():
    from safetune.harden.asrt import ASRTCallback, ASRTConfig

    cb = ASRTCallback(
        attacker=_AlwaysComplies(),
        adversarial_prompts=["p"] * 10,
        config=ASRTConfig(eval_steps=1, probes_per_eval=10, max_buffer=3),
    )
    cb.on_step_end(None, None, None)
    assert len(cb.buffer) == 3  # trimmed to max_buffer
