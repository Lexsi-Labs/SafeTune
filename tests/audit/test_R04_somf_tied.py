"""Audit R04 (review of #109): somf_merge must merge a tied weight once.

embed_tokens and lm_head are one tensor listed under two names; the merge ran on each, so with
lam the tied weight moved lam * lam (0.25 for lam=0.5) instead of lam. Also present on main.
"""
import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from safetune.recover import somf_merge


def test_somf_merge_tied_embedding_moves_exactly_lam():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True)
    aligned = LlamaForCausalLM(cfg).eval()
    ft = copy.deepcopy(aligned)
    with torch.no_grad():
        ft.get_input_embeddings().weight.add_(1.0)           # task vector = +1 on the tied weight
    before = ft.get_input_embeddings().weight.detach().clone()
    masks = {n: torch.ones_like(p) for n, p in ft.state_dict().items()}

    somf_merge(ft, aligned=aligned, lam=0.5, subspace_mask=masks)

    moved = before - ft.get_input_embeddings().weight.detach()   # ft -> aligned + lam * delta
    assert torch.allclose(moved, torch.full_like(moved, 0.5), atol=1e-5), (
        f"tied embedding moved {moved.mean().item():.3f} toward aligned, expected 0.5"
    )


@pytest.mark.parametrize("masks", [{"lm_head.weight": False},
                                   {"model.embed_tokens.weight": False},
                                   {"model.embed_tokens.weight": False, "lm_head.weight": True}])
def test_somf_merge_finds_a_tied_mask_under_any_name(masks):
    # A hand-built all-zero mask reverts the tied weight to aligned, whichever name it is keyed
    # under. With both names keyed, the first name in the state dict wins.
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True)
    aligned = LlamaForCausalLM(cfg).eval()
    ft = copy.deepcopy(aligned)
    with torch.no_grad():
        ft.get_input_embeddings().weight.add_(1.0)
    shape = aligned.get_input_embeddings().weight.shape

    somf_merge(ft, aligned=aligned, subspace_mask={n: torch.full(shape, v) for n, v in masks.items()})

    assert torch.equal(ft.get_input_embeddings().weight, aligned.get_input_embeddings().weight)
