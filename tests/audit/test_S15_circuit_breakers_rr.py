"""Audit S15: CircuitBreakerRRModel installs its hooks on the model, so two wrappers (e.g. two calibrations of the runner trainer)
stacked. Installation now replaces the hooks an earlier wrapper left (same pattern as AdaSteer, commit 2c2c4c2)."""
import safetune.steer as S

from _steer_common import HARMFUL, HARMLESS, build


def _wrapper(m, tok):
    _, _, dirs = S.extract_refusal_direction(m, tok, HARMFUL, HARMLESS, S.RefusalDirectionConfig(select_directions=False))
    return S.CircuitBreakerRRModel(m, directions=dirs, config=S.CircuitBreakerRRConfig(threshold=0.5)).install()


def test_two_wrappers_leave_one_wrappers_worth_of_hooks():
    m, tok = build()
    count = lambda: sum(len(l._forward_hooks) for l in m.model.layers)
    w1 = _wrapper(m, tok)
    one = count()
    w2 = _wrapper(m, tok)
    two = count()
    w2.remove()
    left = count()
    w1.remove()
    assert one == two == 6, (one, two)
    assert left == 0, left


def test_without_baselines_install_warns_once_that_outputs_depend_on_call_history():
    import pytest
    m, tok = build()
    _, _, dirs = S.extract_refusal_direction(m, tok, HARMFUL, HARMLESS, S.RefusalDirectionConfig(select_directions=False))
    w = S.CircuitBreakerRRModel(m, directions=dirs, config=S.CircuitBreakerRRConfig(threshold=0.5))
    with pytest.warns(UserWarning, match="depend on call history") as rec:
        w.install()
        w.install()
    assert len([r for r in rec if "call history" in str(r.message)]) == 1
    w.remove()


def test_with_baselines_there_is_no_warning():
    import warnings
    m, tok = build()
    _, _, dirs = S.extract_refusal_direction(m, tok, HARMFUL, HARMLESS, S.RefusalDirectionConfig(select_directions=False))
    baselines = {l: (0.0, 1.0) for l in dirs}
    w = S.CircuitBreakerRRModel(m, directions=dirs, baselines=baselines, config=S.CircuitBreakerRRConfig(threshold=0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        w.install()
    w.remove()


def test_the_hooks_change_the_output_against_the_unsteered_model():
    import torch
    from _steer_common import prompt_ids
    m, tok = build()
    ids = prompt_ids(tok, HARMFUL[0])
    with torch.no_grad():
        plain = m(input_ids=ids).logits
    w = _wrapper(m, tok)
    with torch.no_grad():
        steered = m(input_ids=ids).logits
    w.remove()
    assert (steered - plain).abs().max() > 1e-3
