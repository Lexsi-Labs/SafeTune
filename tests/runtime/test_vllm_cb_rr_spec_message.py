"""extract_steer_spec must not call CircuitBreakerRRModel a weight-space method (issue #94).

Its runner calibrate path installs runtime hooks; it produces no trained checkpoint.
"""
import pytest
import torch.nn as nn

from safetune.interventions.steer import CircuitBreakerRRModel
from safetune.interventions.steer.backends.vllm_hook import extract_steer_spec


def test_cb_rr_spec_error_points_to_hook_backend_not_checkpoint():
    with pytest.raises(NotImplementedError) as exc:
        extract_steer_spec(CircuitBreakerRRModel(nn.Linear(2, 2)))
    msg = str(exc.value)
    assert "checkpoint" not in msg and "weight-space" not in msg
    assert "build_vllm_eval_backend" in msg
