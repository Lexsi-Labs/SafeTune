"""Audit S09 / S15 / S18: arguments that are accepted and then ignored now warn (once each) when a non-default value is passed."""
import warnings

import pytest
import torch

import safetune.steer as S
from safetune.runner import steer as R

from _steer_common import build


def _warns(match, fn):
    with pytest.warns(UserWarning, match=match) as rec:
        fn()
    return [w for w in rec if match in str(w.message)]


def test_tar_model_warns_for_target_modules():
    assert len(_warns("target_modules is accepted but ignored", lambda: S.TARModel(torch.nn.Linear(2, 2), target_modules=["q_proj"], warn=False))) == 1


def test_tar_model_is_silent_when_target_modules_is_default():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        S.TARModel(torch.nn.Linear(2, 2), warn=False)


@pytest.mark.parametrize("name", ["rr_layers", "target_layers"])
def test_circuit_breaker_rr_trainer_warns_for_each_ignored_layer_argument(name):
    m, tok = build()
    assert len(_warns(f"{name} is accepted but ignored", lambda: R.CircuitBreakerRRTrainer(m, tok, **{name: [1, 2]}))) == 1


def test_circuit_breaker_rr_trainer_is_silent_by_default():
    m, tok = build()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        R.CircuitBreakerRRTrainer(m, tok)


def test_rrfa_trainer_warns_for_member_models():
    m, tok = build()
    assert len(_warns("member_models is accepted but ignored", lambda: R.RRFAEnsembleTrainer(m, tok, member_models=[m]))) == 1
