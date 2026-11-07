"""
Contrastive Decoding (O'Brien & Lewis, "Contrastive Decoding Improves Reasoning in Large Language
Models", arXiv:2309.09117; after Li et al., arXiv:2210.15097).

At each generation step this computes

    final_logits = strong_logits - alpha * weak_logits

on raw logits (a per-row constant against log-probabilities, which the softmax removes): the expert's score
minus ``alpha`` times the amateur's, as in Li et al. with a coefficient. O'Brien & Lewis use
``(1 + beta) * expert - beta * amateur`` (beta = 0.5); that form is not what this module computes.

with an optional adaptive plausibility constraint: only tokens whose
strong-model probability is at least ``adaptive_eps * max_prob`` are
considered candidates; others are masked to -inf.

For SafeTune we use this to flip the sign-convention: "strong" = the aligned
target model, "weak" = the misaligned / pre-FT base. The resulting decoding
emphasizes tokens the target model prefers and the weak model would not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from ._base import GuidedLogitsProcessor


@dataclass
class ContrastiveDecodingConfig:
    """Configuration for Contrastive Decoding.

    Attributes:
        alpha: contribution of the weak-model subtraction term. ``0.5`` is the
            paper default; smaller values are gentler.
        adaptive_eps: plausibility threshold relative to argmax probability;
            tokens with target-model probability < ``eps * max_prob`` are
            masked to -inf. ``0.1`` is the paper default. Set to ``0`` to disable.
    """

    alpha: float = 0.5
    adaptive_eps: float = 0.1


class ContrastiveDecodingProcessor(GuidedLogitsProcessor):
    """``LogitsProcessor`` implementing Contrastive Decoding."""

    def __init__(
        self,
        guide: Any,
        tokenizer_target: Any,
        tokenizer_guide: Optional[Any] = None,
        config: Optional[ContrastiveDecodingConfig] = None,
        *,
        target: Optional[Any] = None,
    ) -> None:
        super().__init__(guide=guide, tokenizer_target=tokenizer_target, tokenizer_guide=tokenizer_guide,
                         target=target)
        self.config = config or ContrastiveDecodingConfig()

    def combine(
        self,
        scores: torch.Tensor,
        guide_scores: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        target = scores.float()
        # Replace -inf in guide (untranslated tokens) with 0 so subtraction is a no-op there.
        finite_guide = torch.where(torch.isinf(guide_scores), torch.zeros_like(guide_scores), guide_scores)
        # `target - alpha*guide`: the expert minus a scaled amateur (Li et al. 2210.15097 with a
        # coefficient). O'Brien & Lewis 2309.09117 use `(1+beta)*target - beta*guide`; that form is
        # not computed here.
        out = target - cfg.alpha * finite_guide

        if cfg.adaptive_eps > 0:
            target_probs = torch.softmax(target, dim=-1)
            max_p = target_probs.max(dim=-1, keepdim=True).values
            allowed = target_probs >= (cfg.adaptive_eps * max_p)
            out = torch.where(allowed, out, torch.full_like(out, float("-inf")))
        return out.to(scores.dtype)


__all__ = ["ContrastiveDecodingProcessor", "ContrastiveDecodingConfig"]
