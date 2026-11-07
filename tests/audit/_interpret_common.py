"""Shared fixtures for the Interpret audit repros: a random tiny Llama + the tiny test tokenizer.
Skips (never fails) when the Hub is unreachable and the tokenizer is not cached. Benign placeholder prompts only."""
import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
HARMFUL = ["PLACEHOLDER secret request number %d please" % i for i in range(6)]
HARMLESS = ["PLACEHOLDER ordinary request number %d please" % i for i in range(6)]


def build(layers=4, hidden=64, seed=0, padding_side=None):
    try:
        tok = AutoTokenizer.from_pretrained(TINY)
    except OSError as e:
        pytest.skip(f"{TINY} tokenizer is not available offline: {type(e).__name__}")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if padding_side:
        tok.padding_side = padding_side
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=32000, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256)
    return LlamaForCausalLM(cfg).eval(), tok
