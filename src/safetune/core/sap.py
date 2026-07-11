"""
Safety-Aware Probing (SAP) Optimization Framework.
arXiv:2505.16737 - Mitigating Fine-tuning Risks in LLMs via Safety-Aware Probing Optimization.

SAP uses a bi-level optimization approach to mitigate safety degradation during
fine-tuning. It identifies a "safety-aware probe" (a small perturbation V) that
maximizes a Safe-Useful loss, encouraging the model to favor safe updates
within the fine-tuning step.
"""

import logging
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SAPOptimizer:
    """
    Safety-Aware Probing (SAP) Optimizer.
    
    Wraps an existing model and optimizer to inject bi-level SAP steps.
    Requires a callable that computes the `safe_useful_loss` given the model.
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        target_layers: list = None,
    ):
        """
        Args:
            model: The LLM to train.
            base_optimizer: The base optimizer (e.g., AdamW) for W.
            rho: Perturbation magnitude limit for the probe V.
            target_layers: List of parameter names to perturb (if None, all).
        """
        self.model = model
        self.base_optimizer = base_optimizer
        self.rho = rho
        # Store initial state for the probe
        self.base_params = {
            n: p for n, p in model.named_parameters() if p.requires_grad
        }
        
        if target_layers is not None:
            self.target_params = {
                n: p for n, p in self.base_params.items()
                if any(t in n for t in target_layers)
            }
        else:
            self.target_params = self.base_params

    @torch.no_grad()
    def _apply_probe(self, grads: dict[str, torch.Tensor]) -> None:
        """Add probe V to weights: W + V."""
        norm = torch.norm(
            torch.stack([g.norm(p=2) for g in grads.values()])
        ) + 1e-12
        scale = self.rho / norm
        
        for n, p in self.target_params.items():
            if n in grads and grads[n] is not None:
                # V = rho * grad / norm
                e_w = grads[n] * scale
                p.add_(e_w)
                self._current_probes[n] = e_w

    @torch.no_grad()
    def _revert_probe(self) -> None:
        """Remove probe V from weights: W - V."""
        for n, p in self.target_params.items():
            if n in self._current_probes:
                p.sub_(self._current_probes[n])
        self._current_probes = {}

    def step(
        self,
        compute_loss_fn: Callable[[], torch.Tensor],
        compute_safe_useful_loss_fn: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """
        Perform a single bi-level SAP optimization step.
        
        1. Compute grad of safe_useful_loss w.r.t W to find V.
        2. Perturb model: W' = W + V.
        3. Compute standard task loss on W'.
        4. Revert model: W = W' - V.
        5. Backprop task loss to update W using base_optimizer.
        """
        self._current_probes = {}
        
        # Step 1: Find safe probe direction V using Safe-Useful Loss
        self.base_optimizer.zero_grad()
        su_loss = compute_safe_useful_loss_fn()
        su_loss.backward()
        
        su_grads = {n: p.grad.clone() for n, p in self.target_params.items() if p.grad is not None}
        
        # Step 2: Apply probe V
        self._apply_probe(su_grads)
        self.base_optimizer.zero_grad()
        
        # Step 3: Compute usefulness (task) loss on W+V
        task_loss = compute_loss_fn()
        task_loss.backward()
        
        # Step 4: Revert probe
        self._revert_probe()
        
        # Step 5: Update W
        self.base_optimizer.step()
        
        return task_loss
