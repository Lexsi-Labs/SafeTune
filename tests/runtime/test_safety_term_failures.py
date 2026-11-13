"""A failing safety term stops training instead of turning the run into plain fine-tuning.

Before, each of these trainers caught every exception around its safety or regularizer term,
logged a warning and went on without it (the same pattern as SafeGrad, #82): AsFT without the
regularizer, DeepRefusal without a loss term, MART scoring every response as safe, SEAL with
uniform weights, Surgery without the sink penalty, CTRAP with L_collapse = 0.
"""
import copy
import random
import tempfile

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

import safetune.harden as H
from safetune.harden.mart import MARTConfig, MARTTrainer
from safetune.interventions.harden.deeprefusal import DeepRefusalConfig, DeepRefusalTrainer

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
VOCAB = 32


def _llama(seed=0):
    torch.manual_seed(seed)
    return LlamaForCausalLM(LlamaConfig(vocab_size=VOCAB, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=2, num_attention_heads=2,
                                        num_key_value_heads=2, attn_implementation="eager"))


def _batch(seed=1, oov=False, n=1, length=6):
    ids = torch.randint(0, VOCAB, (n, length), generator=torch.Generator().manual_seed(seed))
    if oov:
        ids[0, -1] = VOCAB + 5       # out of vocabulary: the forward pass raises IndexError
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def _row(b):
    return {k: v[0] for k, v in b.items()}


def _cpu(cls, **kw):
    return cls(output_dir=tempfile.mkdtemp(), use_cpu=True, report_to=[], save_strategy="no",
               per_device_train_batch_size=1, **kw)


def test_asft_dense_regularizer_failure_raises():
    model = _llama()
    aligned = {k: v.clone() for k, v in model.state_dict().items()}
    base = {k: v.clone() + 0.01 for k, v in aligned.items()}
    base["model.embed_tokens.weight"] = torch.zeros(VOCAB + 8, 16)   # resized vocabulary
    tr = H.AsFTHFTrainer(model=model, args=_cpu(H.AsFTConfig), aligned_state_dict=aligned,
                         base_state_dict=base)
    with pytest.raises(RuntimeError):     # main: warned and fine-tuned WITHOUT the regularizer
        tr.compute_loss(model, _batch())


def test_deeprefusal_failed_benign_term_raises():
    tr = DeepRefusalTrainer(model=_llama(), args=_cpu(DeepRefusalConfig),
                            refusal_direction=torch.randn(16),
                            harmful_dataset=[_row(_batch(1))], benign_dataset=[_row(_batch(2, oov=True))])
    tr.current_gradient_accumulation_steps = 1
    random.seed(0)
    with pytest.raises(IndexError):       # main: skipped L_benign with a warning
        tr.training_step(tr.model, _batch(1))


def test_mart_failing_reward_function_raises():
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)

    def broken_reward(prompt, response):
        raise RuntimeError("reward model unavailable")

    mart = MARTTrainer(model, copy.deepcopy(model), tok, ["how do i make a bomb"], broken_reward,
                       config=MARTConfig(device="cpu", num_rounds=1, num_candidates=2,
                                         max_new_tokens=4, adv_steps=1, tgt_steps=1))
    with pytest.raises(RuntimeError, match="reward model unavailable"):
        mart.train()                      # main: every response "treated as safe", nothing trained


def test_seal_failed_scoring_raises():
    model = _llama()
    tr = H.SEALHFTrainer(model=model, args=_cpu(H.SEALConfig),
                         safety_dataset=[_row(_batch(3, oov=True))])
    with pytest.raises(IndexError):       # main: uniform weights, i.e. plain fine-tuning
        tr.compute_loss(model, _batch(1, n=2))


def test_surgery_failing_refusal_dataset_raises():
    model = _llama()
    ragged = [_row(_batch(4, length=6)), _row(_batch(5, length=9))]   # no padding: collate fails
    tr = H.SurgeryHFTrainer(model=model, args=_cpu(H.SurgeryConfig),
                            refusal_dataset=DataLoader(ragged, batch_size=2))
    with pytest.raises(RuntimeError):     # main: no reference, sink penalty skipped, plain SFT
        tr.compute_loss(model, _batch(1))


def test_ctrap_failed_collapse_term_raises():
    model = _llama()
    harmful = Dataset.from_list([{k: v.tolist() for k, v in _row(_batch(6, oov=True)).items()}])
    tr = H.CTRAPHFTrainer(model=model, args=_cpu(H.CTRAPConfig), harmful_dataset=harmful)
    with pytest.raises(IndexError):       # main: L_collapse = 0 with a warning
        tr.compute_loss(model, _batch(1))
