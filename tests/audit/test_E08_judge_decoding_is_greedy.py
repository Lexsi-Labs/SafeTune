"""Audit E08: evaluate() decodes the judge greedily, whatever the judge's generation_config says.

_run_judge_batch called judge_model.generate() without do_sample, so decoding followed the judge
model's own generation_config. The three default judges ship a greedy config, but a judge set with
configure(judges={...}) whose config samples (most chat models: Llama-3.1-Instruct has
do_sample=True, temperature=0.6) gave a different verdict for the same input on every call.
run_judge() and the target generation in evaluate() both pass do_sample=False already.
"""
import importlib

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import pytest

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_judge_output_does_not_depend_on_the_seed():
    ev = importlib.import_module("safetune.instrumentation.evaluate.suite.evaluate")
    try:
        tok = AutoTokenizer.from_pretrained(TINY)
    except OSError as e:
        pytest.skip(f"{TINY} tokenizer not available offline: {type(e).__name__}")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    judge = LlamaForCausalLM(LlamaConfig(vocab_size=32000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=4)).eval()
    judge.generation_config.do_sample = True          # a chat model's default
    judge.generation_config.temperature = 1.5
    judge.generation_config.pad_token_id = tok.pad_token_id

    outs = set()
    for seed in range(6):
        torch.manual_seed(seed)
        outs.add(tuple(ev._run_judge_batch(tok, judge, ["Is this response harmful?"], max_new_tokens=4)))
    assert len(outs) == 1, f"{len(outs)} different judge outputs for one input"
