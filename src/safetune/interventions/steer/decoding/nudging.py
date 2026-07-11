"""
Nudging (Fei et al., 2024) — "Nudging: Inference-time Alignment of LLMs via
Guided Decoding" (arXiv:2410.09300). Original repo: https://github.com/fywalter/nudging

Conditional decoding: on tokens where the base (target) model is *uncertain*,
defer to a small aligned guide; on confident steps, take the target's
prediction unchanged.

Authors' algorithm
------------------
The nudging criterion (``check_need_nudging`` in the authors' ``utils.py``,
``nudging_method='top_prob'``) is:

    base_top_prob = exp(max logprob of the base model's next-token dist)
    need_nudging  = base_top_prob < top_prob_thres        # default 0.3

When ``need_nudging`` is true the small *aligned* model generates the next
(nudging) tokens; once the base model regains confidence (top-1 probability
>= threshold) control returns to the base model. The uncertainty statistic
is the base model's **top-1 token probability** — NOT the entropy of the
full distribution — and the default threshold is ``0.3``.

Faithfulness note
-----------------
This module is a single-step ``transformers.LogitsProcessor``: each call sees
exactly one decoding step. The authors' phrase-level dynamic (the guide
*autoregressively generates* a multi-token nudging phrase, then hands control
back) cannot be reproduced inside a per-step logit hook — that is a framework
constraint, not a modelling choice. What *is* faithfully reproduced here is
the authors' decision rule: per step, hand the step over to the aligned guide
exactly when the base model's top-1 probability is below ``top_prob_thres``,
and take the target otherwise. Because HF ``generate`` calls the processor on
every step, deferring to the guide on every uncertain step yields the same
intervention set as the paper (the guide drives the uncertain tokens), just
without an explicit "phrase" grouping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from ._base import GuidedLogitsProcessor


@dataclass
class NudgingConfig:
    """Configuration for Nudging.

    Attributes:
        top_prob_thres: hand the step over to the aligned guide when the base
            model's top-1 next-token probability is *below* this value
            (``base_top_prob < top_prob_thres``). This is the authors'
            ``--top_prob_thres`` / ``thresholds['top_prob']``; their default
            is ``0.3`` (run scripts also use ``0.4``). Higher values nudge
            more often.
        soft_blend: if True, instead of a hard switch, blend the two logit
            distributions by ``alpha = sigmoid((top_prob_thres - p1) / temp)``
            where ``p1`` is the base top-1 probability — i.e. ``alpha -> 1``
            (full guide) when the base is uncertain, ``alpha -> 0`` when it is
            confident. This is a smoothing convenience and is NOT part of the
            authors' algorithm (their switch is hard); kept off by default.
        soft_blend_temp: temperature of the soft-blend sigmoid. Smaller =
            sharper, closer to the hard switch.
        entropy_threshold: deprecated. The original module used full-distribution
            Shannon entropy as the uncertainty statistic; the paper uses top-1
            probability. Retained only so existing callers do not break; it is
            ignored. Use ``top_prob_thres`` instead.
    """

    top_prob_thres: float = 0.3
    soft_blend: bool = False
    soft_blend_temp: float = 0.1
    # Magnitude of the nudge toward the guide on a hand-over step, in logit
    # space: ``out = target + nudge_strength * (guide - target)``. ``1.0``
    # reproduces the authors' hard switch (full hand-over); ``<1`` softens the
    # nudge, ``>1`` amplifies it.
    nudge_strength: float = 1.0
    # Optional token-id biasing applied to EVERY step's output logits:
    # ``safe_tokens`` are boosted by ``safe_token_boost``; ``unsafe_tokens`` are
    # masked to ``-inf`` (never emitted). Both default to no biasing.
    safe_tokens: Optional[List[int]] = None
    unsafe_tokens: Optional[List[int]] = None
    safe_token_boost: float = 5.0
    # Deprecated / ignored — kept for backward compatibility of the kwarg name.
    entropy_threshold: Optional[float] = None


class NudgingProcessor(GuidedLogitsProcessor):
    """``LogitsProcessor`` implementing top-1-probability-conditional Nudging."""

    def __init__(
        self,
        guide: Any,
        tokenizer_target: Any,
        tokenizer_guide: Optional[Any] = None,
        config: Optional[NudgingConfig] = None,
    ) -> None:
        super().__init__(guide=guide, tokenizer_target=tokenizer_target, tokenizer_guide=tokenizer_guide)
        self.config = config or NudgingConfig()

    @staticmethod
    def _top1_prob(logits: torch.Tensor) -> torch.Tensor:
        """Top-1 token probability of a logit row — the paper's nudging signal.

        Matches ``check_need_nudging`` (``nudging_method='top_prob'``):
        ``base_top_prob = exp(max logprob)``. Shape: ``(batch,)``.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.max(dim=-1).values.exp()

    def combine(
        self,
        scores: torch.Tensor,
        guide_scores: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        target = scores.float()
        # Tokens with no guide counterpart arrive as -inf; fall back to the
        # target there so the guide never proposes an untranslatable token.
        guide = torch.where(torch.isinf(guide_scores), target, guide_scores.float())

        # Uncertainty signal: base model's top-1 next-token probability.
        p1 = self._top1_prob(target)  # shape (batch,)

        if cfg.soft_blend:
            # alpha -> 1 (guide) when base is uncertain (p1 below threshold),
            # alpha -> 0 (target) when base is confident. Not in the paper;
            # a smoothing convenience around the hard switch below.
            temp = max(cfg.soft_blend_temp, 1e-6)
            alpha = torch.sigmoid((cfg.top_prob_thres - p1) / temp).unsqueeze(-1)
            coeff = alpha * cfg.nudge_strength
        else:
            # Hard switch (the authors' rule): hand over to the aligned guide on
            # uncertain steps, i.e. when base_top_prob < top_prob_thres.
            hand_over = (p1 < cfg.top_prob_thres).unsqueeze(-1).float()
            coeff = hand_over * cfg.nudge_strength

        # out = target + coeff * (guide - target). coeff==1 -> full guide
        # (the authors' switch); coeff==0 -> unchanged target.
        out = target + coeff * (guide - target)
        out = self._bias_tokens(out)
        return out.to(scores.dtype)

    def _bias_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """Boost ``safe_tokens`` and mask ``unsafe_tokens`` on the output logits."""
        cfg = self.config
        if cfg.safe_tokens:
            idx = torch.as_tensor(cfg.safe_tokens, device=logits.device, dtype=torch.long)
            logits[..., idx] = logits[..., idx] + cfg.safe_token_boost
        if cfg.unsafe_tokens:
            idx = torch.as_tensor(cfg.unsafe_tokens, device=logits.device, dtype=torch.long)
            logits[..., idx] = float("-inf")
        return logits


__all__ = ["NudgingProcessor", "NudgingConfig"]
