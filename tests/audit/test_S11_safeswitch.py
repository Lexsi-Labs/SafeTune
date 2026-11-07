"""Audit S11: SafeSwitch (Hanpx20/SafeSwitch@96a2c86; src/utils.py:163 `judge_risk`, `>= threshold`)."""
import pytest
import torch

import safetune.steer as S
from safetune.interventions.steer.safeswitch import MLPProber

from _steer_common import build



def _hs():
    return tuple(torch.randn(2, 5, 8) for _ in range(3))


def test_mlp_prober_takes_the_last_real_token_of_a_left_padded_row():
    hs = _hs()
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    feat = MLPProber(8, layer_idx=2)._last_token_feature(hs, mask)
    assert torch.equal(feat[0], hs[2][0, 4]), "left-padded row 0 is pooled at position 2, the last real token is 4"
    assert torch.equal(feat[1], hs[2][1, 4])


def test_right_padded_and_unmasked_pooling_are_unchanged():
    hs = _hs()
    right = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    feat = MLPProber(8, layer_idx=2)._last_token_feature(hs, right)
    assert torch.equal(feat[0], hs[2][0, 2]) and torch.equal(feat[1], hs[2][1, 4])
    assert torch.equal(MLPProber(8, layer_idx=2)._last_token_feature(hs, None), hs[2][:, -1, :])


class _Boom:
    def predict_unsafe_probabilities(self, *a, **k):
        raise RuntimeError("probe failed")


def test_a_failing_prober_raises_instead_of_reading_as_safe():
    m, tok = build()
    w = S.SafeSwitchModel(m, prober=_Boom(), probe_layer=3, hidden_size=64, refusal_head=m.lm_head.weight.detach().clone())
    with pytest.raises(RuntimeError, match="probe failed"):
        w._compute_p_unsafe(torch.tensor([[1, 2, 3]]))
    from safetune.core.runtime.inference.safeswitch import SafeSwitchRunner
    with pytest.raises(RuntimeError, match="probe failed"):
        SafeSwitchRunner(m, _Boom())._check_prompt_safety(torch.tensor([[1, 2, 3]]))


def test_pilot_decoding_unavailable_falls_back_to_stage_one():
    m, tok = build()

    class P:
        def predict_unsafe_probabilities(self, hs, mask=None):
            return [0.3]

    w = S.SafeSwitchModel(m, prober=P(), compliance_prober=P(), probe_layer=3, hidden_size=64, refusal_head=m.lm_head.weight.detach().clone())
    w._pilot_hidden_states = lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("no generate"))
    assert w._compute_p_unsafe(torch.tensor([[1, 2, 3]])) == [0.3]


def test_the_faithful_path_flags_a_prompt_at_exactly_the_threshold():
    m, tok = build()

    class AtThreshold:
        def predict_unsafe_probabilities(self, hs, mask=None):
            return [0.5]

    zero = torch.zeros_like(m.lm_head.weight)
    w = S.SafeSwitchModel(m, prober=AtThreshold(), probe_layer=3, hidden_size=64, unsafe_threshold=0.5, refusal_head=zero)
    ids = torch.tensor([[1, 2, 3]])
    base = m.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=0)
    out = w.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=0)
    assert not torch.equal(out, base), "p_unsafe == threshold must switch to the refusal head (>=, as the official judge_risk)"


class _NaNProber:
    def predict_unsafe_probabilities(self, *a, **k):
        return [float("nan")]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_p_unsafe_raises_like_a_prober_failure(bad):
    m, tok = build()

    class P:
        def predict_unsafe_probabilities(self, *a, **k):
            return [bad]

    w = S.SafeSwitchModel(m, prober=P(), probe_layer=3, hidden_size=64, refusal_head=m.lm_head.weight.detach().clone())
    with pytest.raises(ValueError, match="non-finite"):
        w._compute_p_unsafe(torch.tensor([[1, 2, 3]]))
    from safetune.core.runtime.inference.safeswitch import SafeSwitchRunner
    with pytest.raises(ValueError, match="non-finite"):
        SafeSwitchRunner(m, P())._check_prompt_safety(torch.tensor([[1, 2, 3]]))


def test_finite_probabilities_are_untouched():
    m, tok = build()

    class P:
        def predict_unsafe_probabilities(self, *a, **k):
            return [0.25]

    w = S.SafeSwitchModel(m, prober=P(), probe_layer=3, hidden_size=64, refusal_head=m.lm_head.weight.detach().clone())
    assert w._compute_p_unsafe(torch.tensor([[1, 2, 3]])) == [0.25]
