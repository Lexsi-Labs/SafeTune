"""Shared fixtures for the Evaluate audit repros: a random 6-layer Llama and the tiny test tokenizer.

Everything skips (never fails) when the Hub is unreachable and the tokenizer is not cached.
Benign placeholder strings only; no judge API is ever called.
"""
import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
HARMFUL = ["PLACEHOLDER request %d: do the forbidden thing" % i for i in range(8)]
HARMLESS = ["PLACEHOLDER request %d: do the ordinary thing" % i for i in range(8)]


def build(layers=6, hidden=64, seed=0):
    try:
        tok = AutoTokenizer.from_pretrained(TINY)
    except OSError as e:
        pytest.skip(f"{TINY} tokenizer is not available offline: {type(e).__name__}")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=32000, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256)
    return LlamaForCausalLM(cfg).eval(), tok
