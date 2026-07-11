"""Regression tests for two optim correctness fixes.

These guard two specific bugs that were fixed on this branch:

  1. SafeGrad ``_ContextManager.__exit__`` ``had_user_grad`` guard: on a global
     gradient conflict (dot < 0), a parameter whose ORIGINAL user ``.grad`` was
     None must stay ``.grad is None`` after the surgery context. The pre-fix code
     synthesized a fresh safety-projected gradient for it, causing the optimizer
     to update a parameter the user loss never touched.

  2. LISA ``switch_state`` ``float(sum_drift)`` fix: ``switch_state`` must not
     crash when no parameter has ``requires_grad=True``. The pre-fix code called
     ``sum_drift.item()`` on the Python float ``0.0`` (the accumulation loop never
     ran), raising ``AttributeError: 'float' object has no attribute 'item'``.

The modules are pure-torch, so no heavy deps are imported.
"""

import torch
import torch.nn as nn


def test_safegrad_leaves_untouched_param_grad_none_on_conflict():
    """Regression: SafeGrad ``had_user_grad`` guard.

    Bug: on a global gradient conflict (dot < 0), a parameter whose original
    user ``.grad`` was None was assigned a fresh safety-projected gradient by
    the scatter-back, so the optimizer would update an untouched parameter.
    Fix: such params must remain ``.grad is None`` after the surgery context.
    """
    from safetune.core.optim import SafeGradProjector

    # Two single-element parameters so we can reason about the global dot product.
    p_touched = nn.Parameter(torch.tensor([2.0]))
    p_untouched = nn.Parameter(torch.tensor([3.0]))

    model = nn.Module()
    model.p_touched = p_touched
    model.p_untouched = p_untouched

    projector = SafeGradProjector(model)

    # Capture POSITIVE safety gradients for BOTH params.
    # safety_loss = 0.5 * (p_touched^2 + p_untouched^2)
    #   => d/d p_touched   = p_touched   = +2.0
    #   => d/d p_untouched = p_untouched = +3.0
    safety_loss = 0.5 * (p_touched ** 2).sum() + 0.5 * (p_untouched ** 2).sum()
    projector.capture_safety_gradients(safety_loss)

    # Sanity: both params have a stored safety gradient.
    assert id(p_touched) in projector.safety_grads
    assert id(p_untouched) in projector.safety_grads

    # User loss touches ONLY p_touched, and conflicts with its safety grad
    # (negative dot). user_loss = -(p_touched^2) => d/d p_touched = -2*p_touched = -4.0.
    # p_untouched is absent from this loss, so its user .grad stays None.
    user_loss = -(p_touched ** 2).sum()

    model.zero_grad()
    with projector.apply_gradient_surgery():
        user_loss.backward()
        # Global dot = (user_grad . safety_grad). p_untouched is padded with zero,
        # so dot = (-4.0)*(+2.0) + 0*(+3.0) = -8.0 < 0 -> conflict path fires.

    # The touched param must have a gradient after surgery.
    assert p_touched.grad is not None

    # The crux of the regression: the untouched param's user grad was None, so
    # after surgery it must STILL be None (no synthesized update).
    assert p_untouched.grad is None


def test_lisa_switch_state_no_trainable_params_does_not_crash():
    """Regression: LISA ``switch_state`` ``float(sum_drift)`` fix.

    Bug: when no parameter has ``requires_grad=True`` the drift-accumulation loop
    never runs, leaving ``sum_drift`` as the Python float ``0.0``. The pre-fix
    logging call did ``sum_drift.item()``, raising
    ``AttributeError: 'float' object has no attribute 'item'``.
    Fix: coerce with ``float(sum_drift)`` so the switch completes cleanly.
    """
    from safetune.core.optim import LisaOptimizerWrapper

    # Model whose every parameter is frozen -> zero trainable params at switch.
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    for param in model.parameters():
        param.requires_grad = False

    wrapper = LisaOptimizerWrapper(model, rho=0.1, warmup_steps=0)

    # Starts in "finetune"; switching to a different state exercises the drift
    # loop / logging path. This must not raise AttributeError.
    wrapper.switch_state("alignment")

    assert wrapper.status == "alignment"

    # Switch back the other way to cover the opposite branch too.
    wrapper.switch_state("finetune")
    assert wrapper.status == "finetune"
