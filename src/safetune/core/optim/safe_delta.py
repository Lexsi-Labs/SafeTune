"""
SafeDelta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets.
ICML 2025 — ColinLu50/SafeDelta

SafeDelta computes the "safe delta" — the parameter-space direction between the
original aligned model and a fine-tuned unsafe version — and constrains subsequent
fine-tuning updates to not project onto that unsafe direction.

Key mechanism:
1. Compute safe_delta = theta_aligned - theta_finetuned_unsafe
2. During each update step, project out the component of the gradient
   that points against the safe_delta direction (i.e., toward unsafe space).
3. Only let through gradient components that are orthogonal-to or aligned-with
   the safe_delta direction.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SafeDeltaConfig:
    """Configuration for SafeDelta parameter-modification defense."""
    # Scale coefficient for the projection: 0.0 = no constraint, 1.0 = full projection
    projection_strength: float = 1.0
    # If True, also regularize weight magnitudes toward aligned model (soft anchoring)
    enable_weight_anchoring: bool = False
    # Weight for the soft anchoring penalty (λ in the paper)
    anchor_lambda: float = 1e-3
    # Parameter name filter: only apply SafeDelta to params whose name contains any of these.
    # Empty list means apply to ALL parameters.
    param_filter: List[str] = field(default_factory=list)


class SafeDeltaWrapper:
    """
    Wraps a model and applies the SafeDelta constraint during gradient updates.

    Usage::

        wrapper = SafeDeltaWrapper(model, aligned_state_dict, unsafe_state_dict)
        wrapper.compute_safe_delta()

        for batch in dataloader:
            loss = compute_loss(model, batch)
            loss.backward()
            with wrapper.apply_safe_delta_constraint():
                pass  # gradients are already modified in-place
            optimizer.step()
    """

    def __init__(
        self,
        model: Any,
        aligned_state_dict: Optional[Dict[str, Any]] = None,
        unsafe_state_dict: Optional[Dict[str, Any]] = None,
        config: Optional[SafeDeltaConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or SafeDeltaConfig()
        # Safe delta vectors keyed by parameter name
        self._safe_delta: Dict[str, Any] = {}

        if aligned_state_dict is not None and unsafe_state_dict is not None:
            self.compute_safe_delta(aligned_state_dict, unsafe_state_dict)

    def _matches_filter(self, name: str) -> bool:
        if not self.config.param_filter:
            return True
        return any(f in name for f in self.config.param_filter)

    def compute_safe_delta(
        self,
        aligned_state_dict: Dict[str, Any],
        unsafe_state_dict: Dict[str, Any],
    ) -> None:
        """
        Precompute safe_delta = theta_aligned - theta_finetuned_unsafe for each
        trainable parameter.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("SafeDelta requires PyTorch.")

        self._safe_delta = {}
        for name, param in self.model.named_parameters():
            if not self._matches_filter(name):
                continue
            if name not in aligned_state_dict or name not in unsafe_state_dict:
                continue
            aligned = aligned_state_dict[name].float()
            unsafe = unsafe_state_dict[name].float()
            delta = aligned - unsafe  # direction pointing FROM unsafe TO aligned
            norm = delta.norm()
            if norm.item() > 0:
                self._safe_delta[name] = delta / norm  # normalised safe direction
            else:
                self._safe_delta[name] = delta

        logger.info("SafeDelta computed safe_delta for %d parameters.", len(self._safe_delta))

    def _project_gradient(self, grad: Any, safe_dir: Any) -> Any:
        """
        Project out the component of 'grad' that points away from the safe direction.
        Only the orthogonal and aligned components are kept.
        strength=1.0: fully remove the unsafe component.
        strength=0.5: halve it.
        """
        try:
            import torch
        except ImportError:
            return grad

        safe_dir = safe_dir.to(grad.device).to(grad.dtype)
        flat_grad = grad.view(-1)
        flat_dir = safe_dir.view(-1)

        # Dot product of gradient with safe direction
        dot = torch.dot(flat_grad, flat_dir)

        # If gradient already points "with" the safe direction (dot > 0), no need to project
        if dot.item() >= 0:
            return grad

        # Remove the unsafe component: grad -= strength * dot * safe_dir
        unsafe_component = dot * flat_dir
        projected = flat_grad - self.config.projection_strength * unsafe_component
        return projected.view_as(grad)

    @contextmanager
    def apply_safe_delta_constraint(self) -> Iterator[None]:
        """
        Context manager (call AFTER loss.backward() but BEFORE optimizer.step()).
        Projects all accumulated gradients in-place.
        """
        yield  # gradients have been accumulated by the time we exit

        if not self._safe_delta:
            logger.warning("SafeDelta: safe_delta not computed yet. Call compute_safe_delta() first.")
            return

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if name not in self._safe_delta:
                continue
            param.grad.data = self._project_gradient(param.grad.data, self._safe_delta[name])

    def compute_anchor_penalty(self, aligned_state_dict: Dict[str, Any]) -> Any:
        """
        Optional soft-anchoring regularization term.
        Returns a scalar loss = lambda * sum(||theta - theta_aligned||^2).
        Add this to your main loss before calling backward().
        """
        try:
            import torch
        except ImportError:
            raise ImportError("SafeDelta requires PyTorch.")

        if not self.config.enable_weight_anchoring:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in aligned_state_dict:
                aligned = aligned_state_dict[name].to(param.device).to(param.dtype)
                penalty = penalty + (param - aligned).pow(2).sum()

        return self.config.anchor_lambda * penalty
