"""Audit R03: task_arithmetic must apply the safety vector once per tensor.

theta_new = theta_ft + alpha * (theta_aligned - theta_base)   (Ilharco et al., 2023)
"""
import copy

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from safetune.recover import task_arithmetic


def _tiny(tie: bool) -> LlamaForCausalLM:
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=tie)
    return LlamaForCausalLM(cfg).eval()


def test_task_arithmetic_tied_embedding_gets_vector_once():
    base = _tiny(tie=True)
    aligned = copy.deepcopy(base)
    with torch.no_grad():
        aligned.get_input_embeddings().weight.add_(1.0)  # safety vector = +1 on the tied weight
    ft = copy.deepcopy(base)
    before = ft.get_input_embeddings().weight.detach().clone()

    task_arithmetic(ft, base=base, aligned=aligned, alpha=1.0)

    delta = ft.get_input_embeddings().weight.detach() - before
    assert torch.allclose(delta, torch.ones_like(delta), atol=1e-5), (
        f"tied embedding moved by {delta.mean().item():.3f}, expected 1.0"
    )
