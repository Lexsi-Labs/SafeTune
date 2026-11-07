"""Audit S17: AdaSteerModel (arXiv 2504.09466; MuyuenLP/AdaSteer@2a02b78).

`__call__` reset the adaptive state on every call; a decode step (seq_len 1) cannot recapture the probe
activations, and the steer hook then fell back to the uniform `rd_max` instead of the input's coefficient.
The official code recomputes only on a prefill (`_L > 1`, For_Steering_LlamaModel_adasteer.py:397).
"""
import torch

import safetune.steer as S
from safetune.core.runtime.inference.vector_extraction import SteeringVectorExtractor, VectorExtractionConfig

from _steer_common import HARMFUL, HARMLESS, build, prompt_ids


def _wrapper(m, tok):
    _, _, dirs = S.extract_refusal_direction(m, tok, HARMFUL, HARMLESS, S.RefusalDirectionConfig(select_directions=False))
    ex = SteeringVectorExtractor(m, tok, VectorExtractionConfig(target_layers=[1, 2], pool_method="last_token"))
    a = S.AdaSteerModel(m, rejection_direction=dirs, target_layers=[2, 3, 4], base_multiplier=15.0, rd_probe_layer=1, hd_probe_layer=2)
    return a.fit_adaptive(ex._collect_activations(HARMFUL), ex._collect_activations(HARMLESS))


def test_decode_loop_through_call_uses_the_same_coefficient_as_generate():
    m, tok = build()
    a = _wrapper(m, tok)
    ids = prompt_ids(tok, HARMLESS[0])
    a.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=0)
    via_generate = float(a._rd_coeff[0])
    a._reset_adaptive_state()
    with torch.no_grad():
        o = a(input_ids=ids, use_cache=True)
        pkv = o.past_key_values
        steps = []
        for t in (5, 6, 7):
            o = a(input_ids=torch.tensor([[t]]), past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            steps.append(None if a._rd_coeff is None else float(a._rd_coeff[0]))
    a.remove_hooks()
    assert via_generate != a.config.rd_max
    assert steps == [via_generate] * 3, f"coefficient per decode step {steps}, generate uses {via_generate:.2f} (rd_max {a.config.rd_max})"


def test_generate_is_unchanged():
    m, tok = build()
    a = _wrapper(m, tok)
    ids = prompt_ids(tok, HARMFUL[0])
    out1 = a.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=0)
    out2 = a.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=0)
    a.remove_hooks()
    assert torch.equal(out1, out2)


def test_two_wrappers_leave_one_wrappers_worth_of_hooks():
    m, tok = build()
    a1 = _wrapper(m, tok)
    one = sum(len(l._forward_hooks) for l in m.model.layers)
    a2 = _wrapper(m, tok)
    two = sum(len(l._forward_hooks) for l in m.model.layers)
    a2.remove_hooks()
    left = sum(len(l._forward_hooks) for l in m.model.layers)
    assert one == two == 5, (one, two)
    assert left == 0, left
    a1.remove_hooks()
