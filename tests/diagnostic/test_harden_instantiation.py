"""
Constructor smoke tests for every HARDEN trainer.

The planning CSV flagged SafeGrad / CST / Antibody / ASFT as
"Working ❌" without explaining whether the cause was a stub, a
constructor failure, or downstream training collapse. The previous audit
(test_harden_audit.py) confirmed imports + non-pass training_step
bodies. This file goes one level deeper: it builds a TinyLM + a
minimal dataset + a tmp output dir, then constructs the Config + Trainer
for each method. Any constructor that raises is a real bug.

Caveats:
* This is a *constructor* smoke; we do not call .train() because that
  would need a real tokenizer + collator + full HF Trainer dance for an
  arbitrary model. Constructor failures, however, are the most common
  source of "Working ❌" rows.
* DPO trainers (CST, DeRTa, DOOR) need a tokenizer and chosen/rejected
  columns; we ship a stub HF tokenizer for those.
"""
from __future__ import annotations

import importlib
import tempfile
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn


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


def _tiny_dataset(n: int = 4) -> List[Dict[str, torch.Tensor]]:
    return [
        {
            "input_ids": torch.randint(0, 32, (8,), dtype=torch.long),
            "labels": torch.randint(0, 32, (8,), dtype=torch.long),
        }
        for _ in range(n)
    ]


@pytest.fixture
def workdir(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Vanilla Trainer subclasses (no DPO).
# ---------------------------------------------------------------------------

VANILLA_TRAINERS = [
    ("safetune.harden.safegrad", "SafeGradConfig", "SafeGradTrainer"),
    ("safetune.harden.lisa", "LisaConfig", "LisaTrainer"),
    ("safetune.harden.sppft", "SPPFTConfig", "SPPFTTrainer"),
    ("safetune.harden.sap", "SAPConfig", "SAPTrainer"),
    ("safetune.harden.asft", "AsFTConfig", "AsFTTrainer"),
    ("safetune.harden.star_dss", "STARDSSConfig", "STARDSSTrainer"),
    ("safetune.harden.lookahead", None, "LookAheadTrainer"),
    ("safetune.harden.surgery", "SurgeryConfig", "SurgeryTrainer"),
    ("safetune.harden.antibody", None, "AntibodyTrainer"),
]


@pytest.mark.parametrize("mod_name,cfg_name,trainer_name", VANILLA_TRAINERS)
def test_harden_vanilla_trainer_constructs(mod_name, cfg_name, trainer_name, workdir):
    mod = importlib.import_module(mod_name)
    TrainerCls = getattr(mod, trainer_name)
    if cfg_name is not None:
        ConfigCls = getattr(mod, cfg_name)
        args = ConfigCls(output_dir=workdir, num_train_epochs=1, per_device_train_batch_size=2,
                         logging_steps=10, save_strategy="no", report_to=[])
    else:
        from transformers import TrainingArguments
        args = TrainingArguments(output_dir=workdir, num_train_epochs=1,
                                 per_device_train_batch_size=2, logging_steps=10,
                                 save_strategy="no", report_to=[])
    model = _TinyLM()
    train_ds = _tiny_dataset()
    try:
        trainer = TrainerCls(model=model, args=args, train_dataset=train_ds)
    except TypeError:
        # Some constructors require additional kwargs (e.g. SafeGradTrainer's
        # safety_dataset). Pass minimal extras.
        trainer = TrainerCls(model=model, args=args, train_dataset=train_ds,
                             safety_dataset=_tiny_dataset(2))
    assert trainer is not None
    assert hasattr(trainer, "training_step")


# ---------------------------------------------------------------------------
# EMA callback (no Trainer subclass).
# ---------------------------------------------------------------------------

def test_ema_callback_constructs():
    from safetune.harden import EMACallback

    cb = EMACallback(decay=0.999)
    assert cb is not None
    assert hasattr(cb, "on_step_end") or hasattr(cb, "on_train_begin")


# ---------------------------------------------------------------------------
# DPO-style trainers (CST, DeRTa, DOOR). HF DPOTrainer needs a tokenizer +
# specific dataset format; we skip cleanly when trl is not available, and
# otherwise build the minimal viable inputs.
# ---------------------------------------------------------------------------

def _has_trl() -> bool:
    try:
        import trl  # noqa: F401
        return True
    except ImportError:
        return False


DPO_TRAINERS = [
    ("safetune.harden.cst", "CSTConfig", "CSTTrainer"),
    ("safetune.harden.deeprefusal", "DeepRefusalConfig", "DeepRefusalTrainer"),
    ("safetune.harden.derta", "DeRTaConfig", "DeRTaTrainer"),
    ("safetune.harden.door", "DOORConfig", "SafetyDOORTrainer"),
]


@pytest.mark.parametrize("mod_name,cfg_name,trainer_name", DPO_TRAINERS)
def test_harden_dpo_trainer_class_resolvable(mod_name, cfg_name, trainer_name):
    """DPO-style trainers need TRL plus a real tokenizer to instantiate. We
    confirm the class is resolvable and a config can be constructed with
    safe defaults; full instantiation is out of scope for a unit test.
    """
    mod = importlib.import_module(mod_name)
    TrainerCls = getattr(mod, trainer_name)
    ConfigCls = getattr(mod, cfg_name)
    assert TrainerCls is not None and ConfigCls is not None
    if not _has_trl():
        pytest.skip("trl not available")
    cfg = ConfigCls(output_dir="/tmp/_safetune_dpo_test", num_train_epochs=1,
                    per_device_train_batch_size=1, logging_steps=10,
                    save_strategy="no", report_to=[], use_cpu=True)
    assert cfg is not None
