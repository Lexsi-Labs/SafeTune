"""Tests for optimization-based safety wrappers (SafeGrad, EMA, high-level API)."""

import torch
import torch.nn as nn


def test_safegrad_projects_conflicting_gradient():
    """SafeGrad should remove gradient components that conflict with safety gradients."""
    from safetune.core.optim import SafeGradProjector

    # Single-parameter linear model for a deterministic gradient check
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    x = torch.ones(1, 1)

    # Safety loss: 0.5 * (w * x)^2  -> safety gradient is positive
    safety_out = model(x)
    safety_loss = 0.5 * (safety_out ** 2).sum()

    projector = SafeGradProjector(model)
    projector.capture_safety_gradients(safety_loss)

    # Store safety gradient for later comparison
    safety_grad = next(iter(projector.safety_grads.values())).clone()

    # User loss: -(w * x)^2  -> user gradient is negative, conflicts with safety_grad
    user_out = model(x)
    user_loss = -(user_out ** 2).sum()

    model.zero_grad()
    with projector.apply_gradient_surgery():
        user_loss.backward()

    user_grad_projected = model.weight.grad.detach().clone()

    # After surgery, the dot product between user gradient and safety_grad
    # should be non-negative (conflicting component removed).
    dot = float((user_grad_projected * safety_grad).sum().item())
    assert dot >= -1e-6


def test_ema_wrapper_updates_shadow_params():
    """EMAOptimizerWrapper should move shadow params towards current model params."""
    from safetune.core.optim import EMAOptimizerWrapper

    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    ema = EMAOptimizerWrapper(model, decay=0.5)

    # Shadow params start at 1.0; move model weights to 3.0 and step EMA.
    with torch.no_grad():
        model.weight.fill_(3.0)

    ema.step(model)

    shadow = ema.shadow_params["weight"].detach().cpu()
    # With decay=0.5: shadow = 0.5 * 1.0 + 0.5 * 3.0 = 2.0
    assert torch.allclose(shadow, torch.full_like(shadow, 2.0), atol=1e-6)


def test_build_safety_optim_bundle_creation():
    """build_safety_optim should instantiate the requested wrappers."""
    from safetune.core.optim import (
        SafeGradConfig,
        EMAConfig,
        SafetyOptimConfig,
        build_safety_optim,
    )

    model = nn.Linear(4, 4)

    cfg = SafetyOptimConfig(
        safegrad=SafeGradConfig(enabled=True),
        ema=EMAConfig(enabled=True, decay=0.9),
    )
    bundle = build_safety_optim(model, cfg)

    assert bundle.safegrad is not None
    assert bundle.ema is not None

