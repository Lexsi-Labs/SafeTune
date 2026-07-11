"""Regression tests for the CRITICAL ``model_id`` fix in the runner pillars.

THE BUG
-------
In every runner base class — ``_HardenBase`` (harden.py), ``_RecoverBase``
(recover.py), ``_SteerBase`` (steer.py), ``_UnlearnBase`` (unlearn.py) —
``model_id`` was a *required* keyword-only argument with no default::

    def __init__(self, model=None, ..., *, model_id, ...):  # no default!

This made EVERY concrete trainer impossible to construct without explicitly
passing ``model_id``: ``Trainer(model, tok)`` raised
``TypeError: __init__() missing 1 required keyword-only argument: 'model_id'``.

THE FIX
-------
``model_id`` is now OPTIONAL (defaults to ``None``) and is resolved through a
``_derive_model_id(model_id, model[, tokenizer])`` helper that falls back, in
order, to: the explicit ``model_id`` -> ``tokenizer.name_or_path`` ->
``model.config._name_or_path`` -> the literal string ``"model"``. The result
is always a non-None, non-empty string so downstream output naming works.

These tests FAIL before the fix (TypeError on construction) and PASS after it.

The runner package transitively imports transformers/datasets at import time,
so the module self-skips when those heavy deps are absent and runs on the GPU
box where they are installed.
"""
from __future__ import annotations

import pytest

# The runner package imports transformers + datasets transitively at import
# time. Self-skip when absent (no ML env); run on the GPU box.
pytest.importorskip("transformers")
pytest.importorskip("datasets")

import torch
import torch.nn as nn


# ── Minimal stubs (mirrors tests/diagnostic/test_harden_instantiation.py) ──────

class _TinyLM(nn.Module):
    """nn.Module that quacks like an HF causal LM enough for Trainer.__init__."""

    def __init__(self, vocab: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.config = type("C", (), {
            "vocab_size": vocab,
            "hidden_size": hidden,
            "num_hidden_layers": 2,
            "model_type": "tiny",
            "use_return_dict": True,
            "torch_dtype": torch.float32,
            # No _name_or_path: forces the tokenizer / "model" fallbacks unless
            # a tokenizer with name_or_path is supplied.
        })()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids=None, labels=None, **kw):
        if input_ids is None:
            input_ids = torch.zeros((1, 4), dtype=torch.long)
        h = self.embed_tokens(input_ids.long())
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return type("O", (), {"logits": logits, "loss": loss})

    def get_input_embeddings(self):
        return self.embed_tokens


class _StubTokenizer:
    """No-op tokenizer stub exposing the ``name_or_path`` attribute used by the
    ``_derive_model_id`` fallback chain."""

    def __init__(self, name_or_path: str = "stub-model"):
        self.name_or_path = name_or_path
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.padding_side = "right"

    def __call__(self, *args, **kwargs):
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    def apply_chat_template(self, messages, **kwargs):
        return " ".join(m.get("content", "") for m in messages)


def _model():
    return _TinyLM()


def _tok(name_or_path: str = "stub-model"):
    return _StubTokenizer(name_or_path)


# ───────────────────────────────────────────────────────────────────────────────
# Test 1: each pillar — construct a concrete trainer WITHOUT model_id.
#
# Pre-fix: every one of these raised
#   TypeError: __init__() missing 1 required keyword-only argument: 'model_id'
# Post-fix: construction succeeds and trainer.model_id is a non-empty string.
# ───────────────────────────────────────────────────────────────────────────────

def _assert_valid_model_id(trainer):
    assert trainer is not None
    assert hasattr(trainer, "model_id")
    assert trainer.model_id is not None
    assert isinstance(trainer.model_id, str)
    assert trainer.model_id != ""


# -- Harden pillar (_HardenBase: model, tokenizer) ------------------------------

def test_harden_trainer_constructs_without_model_id():
    from safetune.runner import harden

    # PlainSFTTrainer: simplest harden trainer, no method hyperparams required.
    trainer = harden.PlainSFTTrainer(_model(), _tok())
    _assert_valid_model_id(trainer)


def test_harden_trainer_with_hparams_constructs_without_model_id():
    from safetune.runner import harden

    # LisaTrainer takes method hyperparams (all defaulted); pass a minimal one.
    trainer = harden.LisaTrainer(_model(), _tok(), lisa_rho=0.1)
    _assert_valid_model_id(trainer)


# -- Recover pillar (_RecoverBase: model only) ----------------------------------

def test_recover_trainer_constructs_without_model_id():
    from safetune.runner import recover

    # WiseFTTrainer: model-only recover trainer (aligned_model optional here).
    trainer = recover.WiseFTTrainer(_model(), alpha=0.5)
    _assert_valid_model_id(trainer)


def test_recover_task_arithmetic_constructs_without_model_id():
    from safetune.runner import recover

    trainer = recover.TaskArithmeticTrainer(_model(), alpha=1.0)
    _assert_valid_model_id(trainer)


# -- Steer pillar (_SteerBase: model, tokenizer) --------------------------------

def test_steer_trainer_constructs_without_model_id():
    from safetune.runner import steer

    # STATrainer: no extra hyperparams beyond model/tok.
    trainer = steer.STATrainer(_model(), _tok())
    _assert_valid_model_id(trainer)


def test_steer_trainer_with_hparams_constructs_without_model_id():
    from safetune.runner import steer

    trainer = steer.RefusalDirectionTrainer(_model(), _tok(), alpha=20.0)
    _assert_valid_model_id(trainer)


# -- Unlearn pillar (_UnlearnBase: model only) ----------------------------------

def test_unlearn_trainer_constructs_without_model_id():
    from safetune.runner import unlearn

    # GradientAscentTrainer: model-only unlearn trainer.
    trainer = unlearn.GradientAscentTrainer(_model(), epochs=1)
    _assert_valid_model_id(trainer)


def test_unlearn_rmu_constructs_without_model_id():
    from safetune.runner import unlearn

    trainer = unlearn.RMUTrainer(_model(), layer_id=7, max_num_batches=4)
    _assert_valid_model_id(trainer)


# ───────────────────────────────────────────────────────────────────────────────
# Test 2: when a tokenizer with name_or_path is supplied (and no explicit
# model_id), the derived model_id reflects the tokenizer name.
#
# _derive_model_id fallback order: model_id -> tokenizer.name_or_path ->
# model.config._name_or_path -> "model". The TinyLM stub has no
# config._name_or_path, so the tokenizer name must win.
# ───────────────────────────────────────────────────────────────────────────────

def test_harden_model_id_derived_from_tokenizer_name_or_path():
    from safetune.runner import harden

    trainer = harden.PlainSFTTrainer(_model(), _tok("my-model"))
    assert trainer.model_id == "my-model"


def test_steer_model_id_derived_from_tokenizer_name_or_path():
    from safetune.runner import steer

    trainer = steer.STATrainer(_model(), _tok("my-model"))
    assert trainer.model_id == "my-model"


def test_recover_model_id_falls_back_to_literal_model():
    # _RecoverBase takes model only (no tokenizer arg) and the TinyLM stub has
    # no config._name_or_path, so the final "model" literal fallback applies.
    from safetune.runner import recover

    trainer = recover.WiseFTTrainer(_model())
    assert trainer.model_id == "model"


def test_unlearn_model_id_falls_back_to_literal_model():
    from safetune.runner import unlearn

    trainer = unlearn.GradientAscentTrainer(_model())
    assert trainer.model_id == "model"


# ───────────────────────────────────────────────────────────────────────────────
# Test 3: an explicitly passed model_id is honored across all four pillars.
# ───────────────────────────────────────────────────────────────────────────────

def test_explicit_model_id_is_honored_all_pillars():
    from safetune.runner import harden, recover, steer, unlearn

    h = harden.PlainSFTTrainer(_model(), _tok("ignored"), model_id="custom")
    assert h.model_id == "custom"

    r = recover.WiseFTTrainer(_model(), model_id="custom")
    assert r.model_id == "custom"

    s = steer.STATrainer(_model(), _tok("ignored"), model_id="custom")
    assert s.model_id == "custom"

    u = unlearn.GradientAscentTrainer(_model(), model_id="custom")
    assert u.model_id == "custom"
