"""Audit S07: AlphaSteer (AlphaLab-USTC/AlphaSteer@3cd0d2e).

The official layer takes the steering vector from the LAST VALID token found through the attention mask
(AlphaLlama.py:~80-110, utils/mask_utils.py:7). SafeTune's hook is a plain forward hook: HF passes the mask to a decoder
layer as a keyword argument, so `inputs` holds only the hidden states and the hook silently takes position T-1, which
is a pad token for a right-padded row.
"""
import pytest
import torch

import safetune.steer as S
from safetune.core.runtime.inference.vector_extraction import SteeringVectorExtractor, VectorExtractionConfig

from _steer_common import HARMFUL, HARMLESS, build

LAYERS = [1, 2, 3]


def _model(m, tok, strength):
    ex = SteeringVectorExtractor(m, tok, VectorExtractionConfig(target_layers=LAYERS, pool_method="last_token"))

    def stack(acts):
        out = torch.zeros(acts[LAYERS[0]].shape[0], max(LAYERS) + 1, 64)
        for l in LAYERS:
            out[:, l] = acts[l].float()
        return out

    return S.AlphaSteerModel(m, harmful_activations=stack(ex._collect_activations(HARMFUL)),
                             benign_activations=stack(ex._collect_activations(HARMLESS)), layers=LAYERS, strength=strength)


def _effect(side, impl):
    m, tok = build()
    m.set_attn_implementation(impl)
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for p in ["Hi", HARMFUL[0]]]
    tok.padding_side = side
    e = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    last = -1 if side == "left" else int(e["attention_mask"][0].sum()) - 1
    out = {}
    for strength in (0.0, 300.0):
        a = _model(m, tok, strength)
        with torch.no_grad():
            out[strength] = m(**e).logits[0, last]
        a.remove_hooks()
    return (out[300.0] - out[0.0]).abs().max()


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_right_padded_row_is_steered_from_its_last_real_token(impl):
    left, right = _effect("left", impl), _effect("right", impl)
    assert left > 1e-3, "premise: steering changes the logits"
    assert abs(left - right) < 1e-3 * left + 1e-5, f"steering effect on the last real token: left-padded {left:.4f}, right-padded {right:.4f}"


def test_last_valid_index_mirrors_the_official_mask_utils():
    f = S.AlphaSteerModel._last_valid_index
    dev = torch.device("cpu")
    assert f(None, 2, 5, dev).tolist() == [4, 4]
    m2 = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0]])
    assert f(m2, 3, 5, dev).tolist() == [2, 4, 0]                     # no valid position -> 0
    neg = torch.finfo(torch.float32).min
    add4 = torch.full((2, 1, 5, 5), neg)
    add4[0, 0, -1, :3] = 0.0
    add4[1, 0, -1, 2:] = 0.0
    assert f(add4, 2, 5, dev).tolist() == [2, 4]                      # eager: additive, 0 = valid
    bool4 = torch.zeros(2, 1, 5, 5, dtype=torch.bool)
    bool4[0, 0, -1, :3] = True
    bool4[1, 0, -1, 2:] = True
    assert f(bool4, 2, 5, dev).tolist() == [2, 4]                     # sdpa: boolean, True = valid


def test_left_padded_and_unpadded_steering_are_unchanged():
    """A left-padded row's last real token is the last position: the steering equals the old (mask-free) result."""
    m, tok = build()
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for p in ["Hi", HARMFUL[0]]]
    tok.padding_side = "left"
    e = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    a = _model(m, tok, 300.0)
    with torch.no_grad():
        with_mask = m(**e).logits
        no_mask = m(input_ids=e["input_ids"]).logits[1]            # unpadded-equivalent row 1 (longest prompt)
    a.remove_hooks()
    assert torch.equal(with_mask[1], no_mask)


def test_two_wrappers_leave_one_wrappers_worth_of_hooks():
    m, tok = build()
    count = lambda: sum(len(l._forward_hooks) + len(l._forward_pre_hooks) for l in m.model.layers)
    a1 = _model(m, tok, 1.0)
    one = count()
    a2 = _model(m, tok, 1.0)
    two = count()
    a2.remove_hooks()
    left = count()
    a1.remove_hooks()
    assert one == two == 6, (one, two)
    assert left == 0, left
