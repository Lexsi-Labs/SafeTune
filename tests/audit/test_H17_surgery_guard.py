"""Audit H17 (guard): Surgery must not silently train as plain SFT when attention weights are missing.

With the default sdpa attention transformers returns an EMPTY attentions tuple (not None); the old
`is None` check never fired, so sink_lambda=0.01 gave exactly the loss of sink_lambda=0 and no warning.
"""
import json
import sys
import tempfile

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

import safetune.harden as H

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _rows(tok, texts):
    out = []
    for t in texts:
        e = tok(t, padding="max_length", max_length=24, return_tensors="pt")
        ids = e["input_ids"][0]
        lab = ids.clone()
        lab[:6] = -100
        lab[e["attention_mask"][0] == 0] = -100
        out.append({"input_ids": ids.tolist(), "attention_mask": e["attention_mask"][0].tolist(),
                    "labels": lab.tolist()})
    return Dataset.from_list(out).with_format("torch")


def _train(attn, sink_lambda):
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.pad_token or tok.eos_token
    benign = _rows(tok, [f"user question {i} answer text here for the model" for i in range(8)])
    refusal = _rows(tok, [f"harmful request {i} I cannot help with that request sorry" for i in range(8)])
    torch.manual_seed(0)
    kw = {} if attn is None else {"attn_implementation": attn}
    m = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32, **kw)
    args = H.SurgeryConfig(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=4, max_steps=2,
                           learning_rate=1e-2, logging_steps=1, report_to=[], use_cpu=True,
                           save_strategy="no", sink_lambda=sink_lambda, seed=0)
    tr = H.SurgeryHFTrainer(model=m, args=args, train_dataset=benign, data_collator=default_data_collator,
                            refusal_dataset=DataLoader(refusal, batch_size=4, collate_fn=default_data_collator))
    tr.train()
    return [h["loss"] for h in tr.state.log_history if "loss" in h]


def test_default_sdpa_model_raises_and_says_to_load_eager():
    with pytest.raises(ValueError, match='attn_implementation="eager"'):
        _train(None, 1.0)


def test_eager_model_runs_and_the_penalty_is_computed(monkeypatch):
    from safetune.core.optim.surgery import SurgeryWrapper

    seen = []
    orig = SurgeryWrapper.compute_sink_penalty

    def spy(self, attentions, attention_mask=None):
        out = orig(self, attentions, attention_mask)
        seen.append((len(attentions), float(out)))
        return out

    monkeypatch.setattr(SurgeryWrapper, "compute_sink_penalty", spy)
    losses = _train("eager", 300.0)
    assert all(map(torch.isfinite, map(torch.tensor, losses)))
    assert seen and all(n == 2 for n, _ in seen)         # 2 layers' attention weights reached the penalty


def _jsonl(path, n=8):
    path.write_text("".join(
        json.dumps({"prompt": f"question {i} about cooking?", "response": f"answer {i} is to use more salt"}) + "\n"
        for i in range(n)))
    return str(path)


def _cli_train(monkeypatch, tmp_path):
    from safetune import cli

    data = _jsonl(tmp_path / "train.jsonl")
    monkeypatch.setattr(sys, "argv", [
        "safetune", "train", "--algo", "surgery", "--model", TINY, "--train-dataset", data,
        "--safety-dataset", data, "--batch-size", "2", "--output", str(tmp_path / "out")])
    cli.main()


def test_cli_default_loaded_model_reaches_the_penalty(monkeypatch, tmp_path):
    """No loader sets attn_implementation, so the CLI's model is sdpa: the runner must switch it to eager."""
    from safetune.core.optim.surgery import SurgeryWrapper

    seen = []
    orig = SurgeryWrapper.compute_sink_penalty

    def spy(self, attentions, attention_mask=None):
        out = orig(self, attentions, attention_mask)
        seen.append((len(attentions), float(out)))
        return out

    monkeypatch.setattr(SurgeryWrapper, "compute_sink_penalty", spy)
    _cli_train(monkeypatch, tmp_path)
    assert (tmp_path / "out" / "config.json").exists()
    assert seen and all(n == 2 for n, _ in seen)         # 2 layers' attention weights reached the penalty


def test_cli_left_padded_batches_get_a_real_penalty(monkeypatch, tmp_path):
    """The tiny Llama tokenizer pads on the left and the CLI keeps that side: key 0 is a pad token."""
    from transformers import AutoTokenizer

    from safetune.core.optim.surgery import SurgeryWrapper

    assert AutoTokenizer.from_pretrained(TINY).padding_side == "left"
    seen = []
    orig = SurgeryWrapper.compute_sink_penalty
    monkeypatch.setattr(SurgeryWrapper, "compute_sink_penalty",
                        lambda self, a, m=None: seen.append(orig(self, a, m)) or seen[-1])
    _cli_train(monkeypatch, tmp_path)
    assert seen and any(float(p) > 0 for p in seen), "the penalty was 0 at every step, so this was plain SFT"


def test_runner_still_raises_when_the_switch_to_eager_has_no_effect(monkeypatch, tmp_path):
    """transformers only warns for models that cannot switch at runtime; the attention check still catches it."""
    from transformers import PreTrainedModel

    monkeypatch.setattr(PreTrainedModel, "set_attn_implementation", lambda self, impl: None)
    with pytest.raises(ValueError, match='attn_implementation="eager"'):
        _cli_train(monkeypatch, tmp_path)


@pytest.mark.parametrize("attn", ["eager", None])
def test_hf_trainer_without_refusal_dataset_raises(attn):
    """Without X_r there is no reference sink profile, so the penalty was skipped: plain SFT."""
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.pad_token or tok.eos_token
    kw = {} if attn is None else {"attn_implementation": attn}
    m = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32, **kw)
    args = H.SurgeryConfig(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=4, max_steps=2,
                           report_to=[], use_cpu=True, save_strategy="no")
    with pytest.raises(ValueError, match="refusal_dataset"):
        H.SurgeryHFTrainer(model=m, args=args, train_dataset=_rows(tok, ["a question and answer"] * 4),
                           data_collator=default_data_collator).train()
