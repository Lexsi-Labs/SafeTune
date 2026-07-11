"""
TAR: Tamper-Resistant Safeguards for Open-Weight LLMs.
rishub-tamirisa/tamper-resistance (ICLR 2025)

Protects models against malicious weight modifications by training
safeguards that withstand significant fine-tuning attacks.  The core
idea is meta-learning: during safety training, simulate N steps of
adversarial fine-tuning and backprop through the unrolled optimiser so
that the final weights still refuse harmful queries.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TARConfig:
    """Configuration for Tamper-Resistant Safeguards."""

    inner_steps: int = 4
    inner_lr: float = 2e-5
    outer_lr_multiplier: float = 1.0
    attack_batch_size: int = 4
    safety_weight: float = 1.0
    capability_weight: float = 0.5
    target_modules: Optional[List[str]] = None


class TARWrapper:
    """Meta-learning tamper resistance for open-weight LLMs.

    During each outer step:
      1. Clone parameters → θ'
      2. Simulate `inner_steps` of adversarial fine-tuning on θ'
      3. Compute safety loss on θ' (adversarial state)
      4. Backprop through the unroll to update the original θ
    """

    def __init__(
        self,
        model: Any,
        config: Optional[TARConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or TARConfig()
        self._param_snapshot: Optional[Dict[str, Any]] = None

    # ── parameter filtering ─────────────────────────────────────

    def _target_params(self) -> List:
        """Return parameters that should be tamper-resistant."""
        result = []
        for name, param in self.model.named_parameters():
            if self.config.target_modules is None:
                result.append((name, param))
            elif any(m in name for m in self.config.target_modules):
                result.append((name, param))
        return result

    # ── inner loop (simulated attack) ───────────────────────────

    def simulate_attack(
        self,
        attack_loss_fn: Any,
        attack_data: Any,
    ) -> Dict[str, Any]:
        """Simulate adversarial fine-tuning and return the attacked state."""
        try:
            import torch
        except ImportError:
            raise ImportError("TAR requires PyTorch.")

        # snapshot current params
        target_params = self._target_params()
        theta_prime: Dict[str, Any] = {}
        for name, param in target_params:
            theta_prime[name] = param.data.clone().requires_grad_(True)

        # inner loop
        for step in range(self.config.inner_steps):
            # build a temporary state dict overlay
            grads = {}
            for name, val in theta_prime.items():
                if val.grad is not None:
                    val.grad.zero_()

            loss = attack_loss_fn(theta_prime, attack_data)
            loss.backward(retain_graph=(step < self.config.inner_steps - 1))

            # SGD update
            for name, val in theta_prime.items():
                if val.grad is not None:
                    theta_prime[name] = val - self.config.inner_lr * val.grad
                    theta_prime[name] = theta_prime[name].detach().requires_grad_(True)

        return theta_prime

    # ── outer loss ──────────────────────────────────────────────

    def compute_tar_loss(
        self,
        safety_loss_fn: Any,
        safety_data: Any,
        attacked_params: Dict[str, Any],
        capability_loss_fn: Optional[Any] = None,
        capability_data: Optional[Any] = None,
    ) -> Any:
        """Compute the TAR outer loss on attacked parameters.

        Loss = safety_weight × L_safety(θ') + capability_weight × L_cap(θ)
        """
        try:
            import torch
        except ImportError:
            raise ImportError("TAR requires PyTorch.")

        # safety loss on attacked params
        safety_loss = safety_loss_fn(attacked_params, safety_data)
        total = self.config.safety_weight * safety_loss

        # capability preservation on original params
        if capability_loss_fn is not None and capability_data is not None:
            original_params = {n: p for n, p in self._target_params()}
            cap_loss = capability_loss_fn(original_params, capability_data)
            total = total + self.config.capability_weight * cap_loss

        return total

    # ── snapshot / restore ──────────────────────────────────────

    def snapshot(self) -> None:
        """Save current model state for rollback."""
        self._param_snapshot = {
            n: p.data.clone() for n, p in self.model.named_parameters()
        }

    def restore(self) -> None:
        """Restore model to last snapshot."""
        if self._param_snapshot is None:
            raise RuntimeError("No snapshot to restore from.")
        sd = self.model.state_dict()
        for name, val in self._param_snapshot.items():
            if name in sd:
                sd[name].copy_(val)
        self.model.load_state_dict(sd)
        logger.info("TAR: restored model to snapshot.")

    def verify_tamper_resistance(
        self,
        safety_eval_fn: Any,
        eval_data: Any,
        threshold: float = 0.9,
    ) -> Dict[str, float]:
        """Evaluate tamper resistance: returns safety score and pass/fail."""
        score = safety_eval_fn(self.model, eval_data)
        return {
            "safety_score": float(score),
            "threshold": threshold,
            "passed": float(score) >= threshold,
        }
