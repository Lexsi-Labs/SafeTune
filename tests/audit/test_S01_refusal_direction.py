"""Audit S01: refusal direction vs andyrdt/refusal_direction@9d852fa.

1. Layer label: extraction reads the block OUTPUT, the official code the block INPUT (`generate_directions.py:12-16`),
   so direction[l] is the official direction of layer l+1 (documented, no code change).
2. Ablation used to hook only the block output; the official one hooks the block input and the attn/mlp outputs
   (`hook_utils.py:80-88`).
"""
import torch

import safetune.steer as S
from safetune.interventions.steer.refusal_direction import RefusalDirectionConfig

from _steer_common import HARMFUL, HARMLESS, build, prompt_ids


def _setup():
    m, tok = build()
    d, _, dirs = S.extract_refusal_direction(m, tok, HARMFUL, HARMLESS, RefusalDirectionConfig(select_directions=False))
    return m, tok, d, dirs


def _official_resid_pre_diff(m, tok):
    def acts(ps):
        out = []
        for p in ps:
            hs = m(input_ids=prompt_ids(tok, p), output_hidden_states=True).hidden_states
            out.append(torch.stack([h[0, -1].double() for h in hs[:len(m.model.layers)]]))   # resid_pre of layer l = hs[l]
        return torch.stack(out).mean(0)
    with torch.no_grad():
        return acts(HARMFUL) - acts(HARMLESS)


def test_direction_of_layer_l_is_the_official_direction_of_layer_l_plus_1():
    """Documents the label: layer l here = output of block l = official layer l+1 (resid_pre of block l+1)."""
    m, tok, _, dirs = _setup()
    od = _official_resid_pre_diff(m, tok)
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=0))
    for l in range(len(m.model.layers) - 1):
        assert cos(dirs[l], od[l + 1]) > 0.999


def test_ablation_matches_the_official_component_hooks():
    m, tok, d, _ = _setup()
    L = m.model.layers
    dd = (d / d.norm()).float()
    proj = lambda x: x - (x @ dd).unsqueeze(-1) * dd
    hs = []
    ids = prompt_ids(tok, HARMFUL[0])
    for l in L:
        hs.append(l.register_forward_pre_hook(
            lambda mod, a, kw: (((proj(a[0]),) + tuple(a[1:]), kw) if a else (a, {**kw, "hidden_states": proj(kw["hidden_states"])})),
            with_kwargs=True))
        for sub in (l.self_attn, l.mlp):
            hs.append(sub.register_forward_hook(lambda mod, i, o: ((proj(o[0]),) + tuple(o[1:])) if isinstance(o, tuple) else proj(o)))
    with torch.no_grad():
        official = m(input_ids=ids).logits[0, -1]
    [h.remove() for h in hs]
    with S.RefusalDirectionModel(m, d, mode="ablate"), torch.no_grad():
        ours = m(input_ids=ids).logits[0, -1]
    assert (ours - official).abs().max() < 1e-4, f"max |ablate - official ablate| = {(ours - official).abs().max():.4f}"


def test_steer_mode_is_the_block_output_addition_and_strength_zero_is_identity():
    m, tok, d, _ = _setup()
    ids = prompt_ids(tok, HARMFUL[0])
    dd = (d / d.norm()).float()
    hs = [l.register_forward_hook(lambda mod, i, o: ((o[0] + 2.0 * dd,) + tuple(o[1:])) if isinstance(o, tuple) else o + 2.0 * dd)
          for l in m.model.layers]
    with torch.no_grad():
        manual = m(input_ids=ids).logits
    [h.remove() for h in hs]
    with torch.no_grad():
        base = m(input_ids=ids).logits
        with S.RefusalDirectionModel(m, d, mode="steer", strength=2.0):
            steered = m(input_ids=ids).logits
        with S.RefusalDirectionModel(m, d, mode="ablate", strength=0.0):
            zero_ablate = m(input_ids=ids).logits
    assert torch.equal(steered, manual)
    assert torch.equal(zero_ablate, base)


def test_hooks_are_removed_after_an_exception():
    m, tok, d, _ = _setup()
    try:
        with S.RefusalDirectionModel(m, d, mode="ablate"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert sum(len(l._forward_hooks) + len(l._forward_pre_hooks) + len(l.self_attn._forward_hooks) + len(l.mlp._forward_hooks)
               for l in m.model.layers) == 0
