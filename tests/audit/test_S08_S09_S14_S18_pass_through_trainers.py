"""Audit S08 / S09 / S14 / S18: the four trainers whose Steer wrapper is a pass-through (the method is a training-time loss
elsewhere) warn once, with the same line, when constructed. Their behaviour is unchanged."""
import warnings

import pytest
import torch

from safetune.runner import steer as R
from safetune.runner.steer._base import PASS_THROUGH_WARNING

from _steer_common import HARMFUL, HARMLESS, build, prompt_ids

FOUR = ["RepBendTrainer", "TARSteerTrainer", "CircuitBreakerTrainer", "RRFAEnsembleTrainer"]


@pytest.mark.timeout(900)
@pytest.mark.parametrize("name", FOUR)
def test_the_trainer_warns_once_with_the_shared_line(name):
    m, tok = build()
    with pytest.warns(UserWarning) as rec:
        getattr(R, name)(m, tok)
    msgs = [str(w.message) for w in rec if "evaluates the model as given" in str(w.message)]
    assert msgs == [PASS_THROUGH_WARNING], msgs


@pytest.mark.timeout(900)
@pytest.mark.parametrize("name", FOUR)
def test_calibrating_does_not_warn_a_second_time_and_the_wrapper_still_passes_through(name):
    m, tok = build()
    with pytest.warns(UserWarning):
        trainer = getattr(R, name)(m, tok)
    ids = prompt_ids(tok, HARMFUL[0])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        wrapped, _ = trainer.calibrate(HARMFUL, HARMLESS)
    assert not [w for w in rec if "applies NO inference-time" in str(w.message) or "evaluates the model as given" in str(w.message)]
    with torch.no_grad():
        assert torch.equal(wrapped(input_ids=ids).logits, m(input_ids=ids).logits)


def test_other_trainers_do_not_warn():
    m, tok = build()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        R.CAATrainer(m, tok)
