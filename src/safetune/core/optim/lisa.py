"""
Lazy Safety Alignment (Lisa) Optimization Wrapper.
Reference: "Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning" (NeurIPS 2024)
Source: github.com/git-disl/Lisa

This module provides a standalone, framework-agnostic implementation of the
Lisa bi-state proximal optimization strategy. It prevents harmful fine-tuning
by anchoring the model's weights between a "safety" state and a "utility" state.
"""

import logging
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Dict, Literal

logger = logging.getLogger(__name__)

class LisaOptimizerWrapper:
    """
    Implements the Bi-State Proximal Optimization (BSO) logic from Lisa.

    Rather than hard-coupling to `transformers.Trainer`, this wrapper allows tracking
    the two states (alignment vs finetune) and mathematically applying the proximal
    loss penalty (`rho`) during the backward pass of any generic training loop.
    """

    def __init__(
        self,
        model: nn.Module,
        rho: float = 0.1,
        warmup_steps: int = 100
    ):
        """
        Args:
            model: The PyTorch model to wrap.
            rho: The proximal constraint multiplier (typically 0.1 to 1.0).
            warmup_steps: Number of initial steps to skip applying the proximal penalty
                          to allow initial optimization momentum.
        """
        self.model = model
        self.rho = rho
        self.warmup_steps = warmup_steps
        self.current_step = 0

        self.status = "finetune"  # Starts in utility finetune mode

        # State tracking mapping name -> detached tensor.
        self.alignment_weights: Dict[str, torch.Tensor] = {}
        self.finetune_weights: Dict[str, torch.Tensor] = {}

        self._init_state()

    def _init_state(self):
        """Initializes the baseline weights for both states."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.alignment_weights[name] = param.data.detach().clone()
                self.finetune_weights[name] = param.data.detach().clone()

    def switch_state(self, new_state: Literal["alignment", "finetune"]):
        """
        Switches the active training state and commits the current weights
        to the OUTGOING state's consensus tracker.

        Invariant:
            alignment_weights  = snapshot of model after the last alignment phase.
            finetune_weights   = snapshot of model after the last finetune phase.

        The proximal penalty in each state pulls toward the OTHER state's snapshot,
        so it is critical that each state's tracker is updated when LEAVING that state.

        Bug that was here before: the branches were swapped — leaving alignment wrote
        to finetune_weights (wrong) and leaving finetune wrote to alignment_weights
        (wrong).  The proximal term therefore always pulled toward the initial weights
        instead of the latest consensus, completely defeating the bi-state protection.
        """
        if new_state == self.status:
            return

        sum_drift = 0.0

        if self.status == "alignment":
            # Leaving alignment: commit current weights to alignment_weights so the
            # finetune proximal term can pull toward this safety consensus.
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.alignment_weights[name] = param.data.detach().clone()
                    sum_drift += torch.norm(
                        self.alignment_weights[name] - self.finetune_weights[name]
                    ) ** 2
            # sum_drift may stay a Python float if no param had requires_grad=True
            # (loop never ran), so coerce instead of calling tensor-only .item().
            logger.info(f"Lisa: Switched to finetune mode. Drift to consensus: {float(sum_drift):.4f}")
        else:
            # Leaving finetune: commit current weights to finetune_weights so the
            # alignment proximal term can pull toward this utility consensus.
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.finetune_weights[name] = param.data.detach().clone()
                    sum_drift += torch.norm(
                        self.finetune_weights[name] - self.alignment_weights[name]
                    ) ** 2
            # sum_drift may stay a Python float if no param had requires_grad=True
            # (loop never ran), so coerce instead of calling tensor-only .item().
            logger.info(f"Lisa: Switched to alignment mode. Drift to consensus: {float(sum_drift):.4f}")

        self.status = new_state

    def step(self):
        """Advances the internal step counter (used for warmup logic)."""
        self.current_step += 1

    @contextmanager
    def apply_proximal_penalty(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Context manager that applies the Lisa proximal L2 penalty to the loss
        before backward is called.

        Usage:
            with lisa_wrapper.apply_proximal_penalty(base_loss) as modified_loss:
                modified_loss.backward()
        """
        modified_loss = loss.clone()

        if self.current_step > self.warmup_steps and self.rho > 0:
            penalty = 0.0

            if self.status == "alignment":
                # Pull alignment optimization towards the finetune consensus
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        penalty += (self.rho / 2) * torch.norm(param - self.finetune_weights[name]) ** 2
            else:
                # Pull finetune optimization towards the alignment consensus
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        penalty += (self.rho / 2) * torch.norm(param - self.alignment_weights[name]) ** 2

            modified_loss = modified_loss + penalty

        yield modified_loss
