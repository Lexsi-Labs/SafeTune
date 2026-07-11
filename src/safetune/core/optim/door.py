"""
DOOR / W-DOOR: Dual-Objective Optimization for Refusal.
ICML 2025

Combines robust refusal training with targeted unlearning.
W-DOOR enhances this by weighting critical "refusal words"
(e.g., "I cannot", "refuse") more heavily in the loss.

SafeDPO: Safety-integrated Direct Preference Optimization.
Automatically generates safety preference pairs and applies
DPO with safety-aware margin scaling.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# DOOR / W-DOOR
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DOORConfig:
    """Configuration for Dual-Objective Optimization for Refusal."""

    refusal_weight: float = 1.0
    unlearning_weight: float = 0.5
    use_weighted_refusal: bool = True    # W-DOOR
    refusal_keywords: List[str] = field(default_factory=lambda: [
        "I cannot", "I can't", "I must refuse", "I'm unable",
        "I won't", "not appropriate", "against my guidelines",
    ])
    keyword_boost: float = 3.0
    unlearning_method: str = "gradient_ascent"  # "gradient_ascent", "kl_divergence"


class DOORTrainer:
    """Dual-Objective Optimization for Refusal training.

    Objective 1 (Refusal): Train the model to consistently refuse
    harmful requests with high quality refusals.

    Objective 2 (Unlearning): Suppress harmful knowledge from the
    model's parameters via gradient ascent on harmful completions.
    """

    def __init__(self, config: Optional[DOORConfig] = None) -> None:
        self.config = config or DOORConfig()

    def compute_weighted_refusal_loss(
        self,
        logits: Any,
        labels: Any,
        input_ids: Any,
        tokenizer: Any,
    ) -> Any:
        """Compute W-DOOR weighted refusal loss.

        Tokens in the refusal response that match keywords get
        a higher weight in the cross-entropy loss.
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("DOOR requires PyTorch.")

        # standard CE loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)

        if not self.config.use_weighted_refusal:
            return loss_per_token.mean()

        # compute token weights
        weights = torch.ones_like(loss_per_token)
        for keyword in self.config.refusal_keywords:
            keyword_ids = tokenizer.encode(keyword, add_special_tokens=False)
            if not keyword_ids:
                continue
            # slide a window to find keyword positions
            seq_len = shift_labels.shape[1]
            kw_len = len(keyword_ids)
            for start in range(seq_len - kw_len + 1):
                match = True
                for j, kid in enumerate(keyword_ids):
                    if not (shift_labels[:, start + j] == kid).all():
                        match = False
                        break
                if match:
                    weights[:, start:start + kw_len] = self.config.keyword_boost

        weighted_loss = (loss_per_token * weights).sum() / weights.sum()
        return weighted_loss

    def compute_unlearning_loss(
        self,
        logits: Any,
        labels: Any,
    ) -> Any:
        """Compute unlearning loss (gradient ascent on harmful completions)."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("DOOR requires PyTorch.")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if self.config.unlearning_method == "gradient_ascent":
            # negative CE → ascend on harmful outputs
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )
            return -ce_loss
        else:
            # KL divergence from uniform → make model uncertain about harmful tokens
            uniform = torch.ones_like(shift_logits) / shift_logits.size(-1)
            log_probs = F.log_softmax(shift_logits, dim=-1)
            kl = F.kl_div(log_probs, uniform, reduction="batchmean")
            return -kl

    def compute_door_loss(
        self,
        refusal_logits: Any,
        refusal_labels: Any,
        harmful_logits: Any,
        harmful_labels: Any,
        input_ids: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> Any:
        """Compute combined DOOR loss."""
        try:
            import torch
        except ImportError:
            raise ImportError("DOOR requires PyTorch.")

        # refusal objective
        if self.config.use_weighted_refusal and tokenizer is not None and input_ids is not None:
            refusal_loss = self.compute_weighted_refusal_loss(
                refusal_logits, refusal_labels, input_ids, tokenizer
            )
        else:
            import torch.nn.functional as F
            shift_logits = refusal_logits[..., :-1, :].contiguous()
            shift_labels = refusal_labels[..., 1:].contiguous()
            refusal_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        # unlearning objective
        unlearn_loss = self.compute_unlearning_loss(harmful_logits, harmful_labels)

        total = (
            self.config.refusal_weight * refusal_loss
            + self.config.unlearning_weight * unlearn_loss
        )
        return total


# ═══════════════════════════════════════════════════════════════════
# SafeDPO
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SafeDPOConfig:
    """Configuration for Safety-integrated DPO."""

    beta: float = 0.1
    safety_margin_scale: float = 2.0
    label_smoothing: float = 0.0
    auto_generate_pairs: bool = True
    refusal_templates: List[str] = field(default_factory=lambda: [
        "I cannot assist with this request as it could lead to harm.",
        "I'm sorry, but I must refuse this request. It goes against safety guidelines.",
        "This request asks for content that I'm not able to provide due to safety policies.",
    ])


class SafeDPOFormatter:
    """Generate safety-aware DPO preference pairs."""

    def __init__(self, config: Optional[SafeDPOConfig] = None) -> None:
        self.config = config or SafeDPOConfig()

    def generate_safety_pairs(
        self,
        harmful_prompts: List[str],
        harmful_responses: List[str],
        safe_responses: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Generate DPO pairs: chosen = safe refusal, rejected = harmful."""
        import random
        pairs = []
        for i, (prompt, harmful) in enumerate(zip(harmful_prompts, harmful_responses)):
            if safe_responses and i < len(safe_responses):
                chosen = safe_responses[i]
            else:
                chosen = random.choice(self.config.refusal_templates)

            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": harmful,
                "safety_pair": True,
                "margin_scale": self.config.safety_margin_scale,
            })
        logger.info("SafeDPO: generated %d safety preference pairs.", len(pairs))
        return pairs

    def compute_safe_dpo_loss(
        self,
        policy_chosen_logps: Any,
        policy_rejected_logps: Any,
        ref_chosen_logps: Any,
        ref_rejected_logps: Any,
        margin_scales: Optional[Any] = None,
    ) -> Any:
        """Compute DPO loss with safety-aware margin scaling."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("SafeDPO requires PyTorch.")

        chosen_rewards = self.config.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.config.beta * (policy_rejected_logps - ref_rejected_logps)

        margin = chosen_rewards - rejected_rewards

        # apply safety margin scaling
        if margin_scales is not None:
            margin = margin * margin_scales

        loss = -F.logsigmoid(margin).mean()

        if self.config.label_smoothing > 0:
            loss = (1 - self.config.label_smoothing) * loss + \
                   self.config.label_smoothing * (-F.logsigmoid(-margin).mean())

        return loss
