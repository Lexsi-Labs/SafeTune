"""
High-level Safety Optimization API.

This module provides a small, opinionated surface for wiring together the
core optimization-based safety defenses in safetune:

- SafeGradProjector: gradient surgery to prevent safety forgetting.
- EMAOptimizerWrapper: Exponential Moving Average of model weights.
- LisaOptimizerWrapper: bi-state proximal optimization.
- SafetyAwareProbingWrapper (SAP): safety-aware probing of parameters.
- DynamicSafetyShapingLoss (STAR-DSS): token-level safety shaping loss.

The goal is to give users a single entrypoint that returns ready-to-use
wrappers for common fine-tuning loops (HF Trainer, TRL, or custom PyTorch).
"""

from dataclasses import dataclass, field
from typing import Optional

import torch.nn as nn

from .safegrad import SafeGradProjector
from .ema import EMAOptimizerWrapper
from .lisa import LisaOptimizerWrapper
from .sap import SafetyAwareProbingWrapper
from .star_dss import DynamicSafetyShapingLoss


@dataclass
class SafeGradConfig:
    """Configuration for SafeGrad gradient surgery."""

    enabled: bool = True


@dataclass
class EMAConfig:
    """Configuration for EMA weight tracking."""

    enabled: bool = False
    decay: float = 0.995


@dataclass
class LisaConfig:
    """Configuration for Lisa bi-state proximal optimization."""

    enabled: bool = False
    rho: float = 0.1
    warmup_steps: int = 100


@dataclass
class SAPConfig:
    """Configuration for Safety-Aware Probing (SAP)."""

    enabled: bool = False
    grad_rate: float = 0.1


@dataclass
class StarDSSConfig:
    """Configuration for STAR-DSS dynamic safety shaping loss."""

    enabled: bool = False
    use_kl_penalty: bool = False
    kl_scale: float = 1.0


@dataclass
class SafetyOptimConfig:
    """
    High-level configuration bundling together safety optimization defenses.

    Example (SafeGrad + EMA):

        config = SafetyOptimConfig(
            safegrad=SafeGradConfig(enabled=True),
            ema=EMAConfig(enabled=True, decay=0.999),
        )

    Data contracts (summary):
    - SafeGrad: expects a scalar ``safety_loss`` whose backward pass defines
      safety gradients for all trainable parameters.
    - SAP: uses token-level labels ``(input_ids, attention_mask, chosen_labels,
      rejected_labels)`` as in
      ``SafetyAwareProbingWrapper.compute_contrastive_safety_gradient``.
    - STAR-DSS: expects ``(logits, labels, safety_weights)`` where
      ``logits.shape == (B, S, V)``, ``labels.shape == (B, S)``, and
      ``safety_weights.shape == (B, S)``.
    """

    safegrad: SafeGradConfig = field(default_factory=SafeGradConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    lisa: LisaConfig = field(default_factory=LisaConfig)
    sap: SAPConfig = field(default_factory=SAPConfig)
    star_dss: StarDSSConfig = field(default_factory=StarDSSConfig)


@dataclass
class SafetyOptimBundle:
    """
    Container for instantiated safety optimization components.

    All fields are optional; they are populated based on SafetyOptimConfig.
    """

    safegrad: Optional[SafeGradProjector] = None
    ema: Optional[EMAOptimizerWrapper] = None
    lisa: Optional[LisaOptimizerWrapper] = None
    sap: Optional[SafetyAwareProbingWrapper] = None
    star_dss_loss: Optional[DynamicSafetyShapingLoss] = None


def build_safety_optim(
    model: nn.Module,
    config: Optional[SafetyOptimConfig] = None,
) -> SafetyOptimBundle:
    """
    Build a SafetyOptimBundle for a given model and config.

    This does not modify the model or optimizer; it simply returns wrappers
    you can plug into your loop.

    Typical custom PyTorch loop (SafeGrad + EMA):

        model = ...
        optimizer = torch.optim.AdamW(model.parameters(), lr=...)

        safety_cfg = SafetyOptimConfig(
            safegrad=SafeGradConfig(enabled=True),
            ema=EMAConfig(enabled=True, decay=0.999),
        )
        safety_optim = build_safety_optim(model, safety_cfg)

        for safety_batch, user_batch in dataloader:
            # 1) Capture safety gradients
            model.zero_grad()
            safety_loss = compute_safety_loss(model, safety_batch)
            safety_optim.safegrad.capture_safety_gradients(safety_loss)

            # 2) User loss with gradient surgery
            model.zero_grad()
            user_loss = compute_user_loss(model, user_batch)
            with safety_optim.safegrad.apply_gradient_surgery():
                user_loss.backward()
            optimizer.step()

            # 3) EMA update
            if safety_optim.ema is not None:
                safety_optim.ema.step(model)

        # Swap in EMA weights before saving (if enabled)
        if safety_optim.ema is not None:
            safety_optim.ema.apply_ema_weights(model)
    """
    cfg = config or SafetyOptimConfig()
    bundle = SafetyOptimBundle()

    if cfg.safegrad.enabled:
        bundle.safegrad = SafeGradProjector(model)

    if cfg.ema.enabled:
        bundle.ema = EMAOptimizerWrapper(model, decay=cfg.ema.decay)

    if cfg.lisa.enabled:
        bundle.lisa = LisaOptimizerWrapper(
            model,
            rho=cfg.lisa.rho,
            warmup_steps=cfg.lisa.warmup_steps,
        )

    if cfg.sap.enabled:
        bundle.sap = SafetyAwareProbingWrapper(
            model,
            grad_rate=cfg.sap.grad_rate,
        )

    if cfg.star_dss.enabled:
        bundle.star_dss_loss = DynamicSafetyShapingLoss(
            use_kl_penalty=cfg.star_dss.use_kl_penalty,
            kl_scale=cfg.star_dss.kl_scale,
        )

    return bundle

