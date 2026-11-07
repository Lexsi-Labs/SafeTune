"""Audit S01: `orthogonalize_weights` leaves embed_tokens alone; the official one edits it
(refusal_direction@9d852fa, llama3_model.py:79-84)."""
import copy

import torch

import safetune.steer as S
from safetune.interventions.steer.refusal_direction import RefusalDirectionConfig, orthogonalize_weights

from _steer_common import HARMFUL, HARMLESS, build


def _setup():
    m, tok = build()
    d, _, _ = S.extract_refusal_direction(m, tok, HARMFUL, HARMLESS, RefusalDirectionConfig(select_directions=False))
    return m, tok, d


def test_orthogonalize_weights_also_edits_the_embedding():
    m, tok, d = _setup()
    m2 = copy.deepcopy(m)
    orthogonalize_weights(m2, d)
    dd = (d / d.norm()).float()
    assert (m2.model.embed_tokens.weight @ dd).abs().max() < 1e-5




def test_projections_are_bit_identical_to_the_plain_formula():
    m, tok, d = _setup()
    m2 = copy.deepcopy(m)
    orthogonalize_weights(m2, d)
    dd = (d / d.norm()).to(torch.float32)
    for l1, l2 in zip(m.model.layers, m2.model.layers):
        for name in ("self_attn.o_proj", "mlp.down_proj"):
            w = l1.get_submodule(name).weight.data
            assert torch.equal(l2.get_submodule(name).weight.data, w - torch.outer(dd, dd) @ w)


def test_tied_embeddings_are_edited_once_through_the_shared_tensor():
    from transformers import LlamaConfig, LlamaForCausalLM
    m, tok, d = _setup()
    cfg = LlamaConfig(vocab_size=32000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=True)
    torch.manual_seed(0)
    t = LlamaForCausalLM(cfg).eval()
    assert t.lm_head.weight is t.model.embed_tokens.weight
    before = t.model.embed_tokens.weight.data.clone()
    snaps = orthogonalize_weights(t, d)
    dd = (d / d.norm()).float()
    assert (t.lm_head.weight @ dd).abs().max() < 1e-5          # the shared tensor was edited
    assert torch.equal(t.model.embed_tokens.weight.data, before - torch.outer(before @ dd, dd))   # once, not twice
    from safetune.interventions.steer.refusal_direction import restore_weights
    restore_weights(t, snaps)
    assert torch.equal(t.model.embed_tokens.weight.data, before)


def test_untied_lm_head_is_left_alone():
    m, tok, d = _setup()
    head = m.lm_head.weight.data.clone()
    orthogonalize_weights(m, d)
    assert torch.equal(m.lm_head.weight.data, head)
