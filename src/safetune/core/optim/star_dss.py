"""
STAR-DSS Dynamic Safety Shaping.
Reference: "Safety Trajectory Assessment of Response (STAR) for Dynamic Safety Shaping (DSS)"
Source: github.com/poloclub/star-dss

This module provides a standalone, framework-agnostic implementation of Dynamic
Safety Shaping (DSS). Traditional alignment treats all tokens in a sequence equally.
DSS uses token-level safety assessments (weights) to dynamically reinforce
safe segments of a sequence while suppressing unsafe ones via a modifier on
the standard Cross Entropy loss.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class DynamicSafetyShapingLoss(nn.Module):
    """
    Computes a value-weighted CrossEntropy loss based on the STAR-DSS methodology.

    Instead of passing a simple (B, S) target array to CrossEntropyLoss, this
    computes the unreduced loss per-token, scales each token's gradients by its
    assessed safety weight, and then aggregates the result.
    """

    IGN_INDEX = -100

    def __init__(self, use_kl_penalty: bool = False, kl_scale: float = 1.0):
        super().__init__()
        self.use_kl_penalty = use_kl_penalty
        self.kl_scale = kl_scale
        # Unreduced CE loss so we can apply temporal weights token-by-token
        self.ce_loss_fn = nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.IGN_INDEX
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        safety_weights: torch.Tensor,
        ref_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Model predictions of shape (Batch, Seq, Vocab)
            labels: Ground truth token IDs of shape (Batch, Seq)
            safety_weights: Token-level modifiers of shape (Batch, Seq) assessing safety.
                            1.0 = safe/reinforce, 0.0 = unsafe/suppress.
            ref_logits: Optional reference model logits for KL penalty, shape (Batch, Seq, Vocab)

        Returns:
            Reduced scalar loss tensor.
        """
        B, S, V = logits.shape

        # Shift to align predictions with targets (predict *next* token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # In case the weights are full length, shift them as well
        if safety_weights.shape[1] == S:
            safety_weights = safety_weights[:, 1:].contiguous()

        # 1. Compute per-token Cross Entropy Loss: Shape (B, S-1)
        # Permute logits to (B, V, S-1) for nn.CrossEntropyLoss
        unreduced_ce = self.ce_loss_fn(shift_logits.permute(0, 2, 1), shift_labels)

        # Ensure weights are on the correct device
        safety_weights = safety_weights.to(unreduced_ce.device)

        # 2. Build the valid mask NOW — used by both the CE and KL terms.
        # Positions where shift_labels == -100 are prompt tokens or padding;
        # they must contribute zero to the loss regardless of what follows.
        valid_mask = (shift_labels != self.IGN_INDEX).float()

        # 3. Scale token gradients dynamically by safety priority.
        # Tokens deemed "unsafe" receive near-zero weights, suppressing their learning.
        weighted_loss = unreduced_ce * safety_weights

        # 4. Handle optional KL-Divergence penalty to stay near a reference model.
        if self.use_kl_penalty and ref_logits is not None:
            shift_ref_logits = ref_logits[..., :-1, :].contiguous()

            # Clamp -100 to 0 before gathering into the vocab dimension.
            # shift_labels contains -100 at prompt/padding positions.  Using -100
            # as a vocab index wraps silently to vocab_size-100 (a valid but
            # completely wrong token) and produces garbage KL values at those
            # positions.  Clamping to 0 makes the gather safe; valid_mask below
            # ensures these positions contribute exactly 0 to the final loss.
            safe_labels = shift_labels.clamp(min=0)

            # log p_model and log p_ref, shape (B, S-1, V)
            logps = torch.log_softmax(shift_logits, dim=-1)
            ref_logps = torch.log_softmax(shift_ref_logits, dim=-1)

            # Per-token log-probs at the target token, shape (B, S-1)
            target_logps = logps.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
            ref_target_logps = ref_logps.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)

            # Schulman's low-variance k3 KL estimator (github.com/joschu blog):
            #   kl = ratio - 1 - log(ratio),   ratio = p_ref / p_model
            # With log_ratio = log p_ref - log p_model:
            #   ratio = exp(log_ratio);  kl = ratio - 1 - log_ratio
            # This is >= 0 everywhere (convexity) and has zero gradient w.r.t.
            # log p_model at equality (p_model == p_ref), so it pulls the policy
            # *toward* the reference instead of pushing unsafe tokens to -inf.
            # The previous signed log-ratio (k1 = log p_model - log p_ref) was
            # unbounded below with a constant non-zero gradient at equality.
            log_ratio = ref_target_logps - target_logps
            kl_div = torch.exp(log_ratio) - 1.0 - log_ratio

            # Apply inverted safety weight AND valid_mask so:
            #   - unsafe tokens (weight=0) get full KL suppression
            #   - safe tokens (weight=1) get zero KL term
            #   - prompt/padding positions (valid_mask=0) never enter the loss
            weighted_kl = kl_div * (1.0 - safety_weights) * self.kl_scale * valid_mask
            weighted_loss = weighted_loss + weighted_kl

        # 5. Mask out prompt/padding positions and compute mean over valid tokens.
        final_loss = (weighted_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

        return final_loss
