"""Audit U05 (review of #110): direction check of the 8 FLAT divergences on a tiny random model.

60 steps on one harmful answer and one refusal. Evidence is only "which way does the loss move
the two answers' mean token probability"; it says nothing about real models. The official
UCSC-REAL/FLAT `get_contrastive_loss` (@bf2ef07) has the same signs, so the loss is unchanged and
flat_unlearn warns for the divergences that went the wrong way.
"""
import logging
import re
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from safetune.interventions.unlearn import flat as F

DOCS = Path(__file__).parents[2] / "docs/user-guide/unlearn/flat.md"

PROMPT = [5, 6, 7]


def _batch(answer):
    ids = torch.tensor([PROMPT + answer])
    labels = ids.clone()
    labels[:, : len(PROMPT)] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


HARM, REFUSAL = _batch([10, 11, 12, 13]), _batch([20, 21, 22, 23])


def _model():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=4))


def _probs(m):
    with torch.no_grad():
        return [F._seq_mean_prob(m(input_ids=b["input_ids"]).logits, b["labels"]).item()
                for b in (HARM, REFUSAL)]


def _run(div):
    m = _model()
    before = _probs(m)
    F.flat_unlearn(m, [HARM], [REFUSAL], None, config=F.FLATConfig(
        variant="flat", divergence=div, epochs=60, lr=1e-3, weight_decay=0.0))
    return before, _probs(m)


@pytest.mark.parametrize("div", ["kl", "jeffrey", "pearson", "jensen_shannon", "total_variation"])
def test_stable_divergences_raise_refusal_and_lower_harm(div, caplog):
    with caplog.at_level(logging.WARNING, logger=F.logger.name):
        (h0, r0), (h1, r1) = _run(div)
    assert h1 < h0 and r1 > r0
    assert not [r for r in caplog.records if "direction check" in r.getMessage()]


@pytest.mark.parametrize("div", ["reverse_kl", "squared_hellinger", "neyman"])
def test_wrong_direction_divergences_warn_once(div, caplog):
    with caplog.at_level(logging.WARNING, logger=F.logger.name):
        (h0, r0), (h1, r1) = _run(div)
    assert r1 < r0                                    # the refusal got less likely
    assert (h1 > h0) if div != "neyman" else (h1 < h0)
    msgs = [r for r in caplog.records if "direction check" in r.getMessage() and div in r.getMessage()]
    assert len(msgs) == 1


def test_docs_recommend_only_the_divergences_that_went_the_right_way():
    prefer = re.search(r"Prefer ([^.]+)\.", DOCS.read_text()).group(1)
    assert set(re.findall(r"`(\w+)`", prefer)) == set(F._DIVERGENCES) - set(F._WRONG_DIRECTION)


def test_flat_fdiv_loss_logs_nothing_and_the_docs_say_so(caplog):
    logits, labels = torch.zeros(1, 4, 8), torch.tensor([[1, 2, 3, 4]])
    with caplog.at_level(logging.WARNING, logger=F.logger.name):
        F.flat_fdiv_loss(logits, labels, logits, labels, divergence="reverse_kl")
    assert not caplog.records
    section = " ".join(DOCS.read_text().split("### flat_fdiv_loss()")[1].split())
    assert "the warning comes only from `flat_unlearn`" in section
