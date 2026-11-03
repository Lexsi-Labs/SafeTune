"""Audit H13 (guards): SafetyDOORTrainer must not turn a missing input into a silent loss.

Before: a batch without the DOOR fields returned a zero loss (no training, no message), and a model
with no reference policy and no adapter-disable path used log pi_ref = 0 (a different loss).
SafetyDOORTrainer is reachable only by constructing it directly (the DOORTrainer runner runs its own
loop); the diagnostics only build its config.
"""
import pytest
import torch
from transformers import AutoModelForCausalLM

from safetune.interventions.harden.door import SafetyDOORTrainer

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _trainer(ref_model=None, pure=True):
    t = object.__new__(SafetyDOORTrainer)     # compute_loss only reads these attributes
    t._door_pure_mode, t._door_alpha, t._door_beta = pure, 0.5, 0.5
    t._door_unlearn_w = t._door_refusal_w = 1.0
    t._wdoor_method, t._wdoor_gamma, t._wdoor_tau = "sigmoid", 1.0, 5.0
    t._proxy_reward_model = None
    t.ref_model = ref_model
    return t


def _batch(n=2, length=12):
    ids = torch.randint(5, 200, (n, length))
    lab = ids.clone()
    lab[:, :4] = -100
    return {"refusal_input_ids": ids, "refusal_labels": lab,
            "harmful_input_ids": ids.flip(-1), "harmful_labels": lab.flip(-1)}


def test_missing_door_fields_raise():
    m = AutoModelForCausalLM.from_pretrained(TINY)
    with pytest.raises(ValueError, match="refusal_input_ids"):
        _trainer().compute_loss(m, {"input_ids": torch.ones(2, 4, dtype=torch.long)})


def test_no_reference_and_no_adapter_raises():
    m = AutoModelForCausalLM.from_pretrained(TINY)
    assert not hasattr(m, "disable_adapter")
    with pytest.raises(ValueError, match="reference policy"):
        _trainer(ref_model=None).compute_loss(m, _batch())


def test_with_a_reference_model_the_loss_is_finite():
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_pretrained(TINY)
    ref = AutoModelForCausalLM.from_pretrained(TINY).eval()
    loss = _trainer(ref_model=ref).compute_loss(m, _batch())
    assert torch.isfinite(loss)
