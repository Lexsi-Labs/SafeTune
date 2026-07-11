"""
SafeDecoding (Xu et al., ACL 2024, arXiv:2402.08983).

Faithful re-implementation of the authors' ``safedecoding_lora`` decoding
routine (https://github.com/uw-nsl/SafeDecoding, ``utils/safe_decoding.py``)
inside an HF :class:`~transformers.LogitsProcessor`.

The original algorithm runs an *expert* model (the base model lightly
fine-tuned on a small safety dataset) alongside the original model and, for
the **first ``m`` generation steps only**:

  1. Convert both models' next-token logits to probabilities via
     ``log_softmax`` (then ``exp``).
  2. Build the *sample space*: starting from the top-``num_common_tokens``
     tokens of each model (ranked by descending probability), take the
     intersection; if fewer than ``num_common_tokens`` tokens are shared,
     **widen the rank window by one and retry** until enough common tokens
     are found (or the vocabulary is exhausted). This is an iterative
     expansion, *not* a fixed top-k intersection.
  3. For every token in that common set compute the steered probability
     ``P_n = P_base + alpha * (P_expert - P_base)`` (floored at ``1e-8``),
     i.e. ``(1 - alpha) * P_base + alpha * P_expert``.
  4. ``softmax``-renormalize over the common set and decode.

After step ``m`` the model decodes normally -- the safety intervention is
switched **off** entirely (a constant-``alpha``-for-``m``-steps-then-off
schedule, *not* a per-step decay).

In SafeTune we expose:

* :class:`SafeDecodingConfig`: ``alpha``, the ``first_m`` window,
  ``num_common_tokens``, and a ``top_k`` candidate cap.
* :class:`SafeDecodingProcessor`: HF ``LogitsProcessor`` that runs the guide
  (expert) alongside the target (original) and applies the per-step blend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from ._base import GuidedLogitsProcessor


@dataclass
class SafeDecodingConfig:
    """Configuration for SafeDecoding.

    Mirrors the authors' ``SafeDecoding.__init__`` parameters
    (``alpha``, ``first_m``, ``top_k``, ``num_common_tokens``).

    Attributes:
        alpha: blend weight on the expert (safety guide). The steered
            probability is ``P_base + alpha * (P_expert - P_base)``. The
            authors' default is ``1.0``; SafeTune keeps that default.
        first_m: number of initial generation steps over which SafeDecoding
            is applied. After ``first_m`` steps the guide contribution is
            switched off and the target decodes normally. Paper/repo use a
            small window (repo default ``5``).
        num_common_tokens: target size of the shared sample space. The rank
            window is expanded until at least this many tokens are common to
            both models' rankings. Authors' default ``3``.
        top_k: candidate cap. Each model's logits are pre-restricted to their
            top-``top_k`` before the common-token search, bounding the cost
            of the (otherwise full-vocab) ``argsort``. The authors argsort
            the whole vocabulary; capping at a generous ``top_k`` is
            behaviour-preserving because the expansion window never needs to
            grow beyond a handful of tokens in practice. Set ``<= 0`` to
            disable the cap. Paper uses ``top_k=10`` for display.
        clip_negative_inf: when target and guide tokenizers differ, replace
            ``-inf`` guide logits (tokens with no guide counterpart) with the
            target logit so the blend cannot mask a token the target prefers.
        decay_steps: **deprecated** alias for ``first_m``. Kept for backward
            compatibility; if set to a positive value and ``first_m`` is left
            at its default it is used as the window length. The authors'
            schedule has no decay -- ``alpha`` is constant within the window.
        alpha0: **deprecated** alias for ``alpha``. Kept for backward
            compatibility with the pre-fix decaying-alpha API; if set it is
            copied into ``alpha`` (the authors' schedule has no decay).
    """

    alpha: float = 1.0
    first_m: int = 5
    num_common_tokens: int = 3
    top_k: int = 50
    clip_negative_inf: bool = True
    # Token ids that must never be emitted — masked to ``-inf`` on every step
    # (both inside and outside the SafeDecoding window). Default: no banning.
    banned_tokens: Optional[List[int]] = None
    decay_steps: Optional[int] = None
    alpha0: Optional[float] = None

    def __post_init__(self) -> None:
        # Backward-compat: pre-fix callers passed ``alpha0`` (decaying-alpha
        # API). Alias it onto ``alpha`` when ``alpha`` is left at its default.
        if self.alpha0 is not None and self.alpha == 1.0:
            self.alpha = float(self.alpha0)

    def window(self) -> int:
        """Effective length of the SafeDecoding window (``first_m``)."""
        if self.decay_steps is not None and self.first_m == 5 and self.decay_steps > 0:
            # Backward-compat: an explicit decay_steps stands in for first_m
            # when first_m was left at the default.
            return int(self.decay_steps)
        return int(self.first_m)


class SafeDecodingProcessor(GuidedLogitsProcessor):
    """``LogitsProcessor`` implementing SafeDecoding."""

    def __init__(
        self,
        guide: Any,
        tokenizer_target: Any,
        prompt_length: int,
        tokenizer_guide: Optional[Any] = None,
        config: Optional[SafeDecodingConfig] = None,
    ) -> None:
        super().__init__(guide=guide, tokenizer_target=tokenizer_target, tokenizer_guide=tokenizer_guide)
        self.config = config or SafeDecodingConfig()
        self.prompt_length = int(prompt_length)

    def _common_token_mask(
        self,
        base_logp: torch.Tensor,
        expert_logp: torch.Tensor,
    ) -> torch.Tensor:
        """Build the per-row boolean mask of the SafeDecoding sample space.

        Replicates the authors' iterative rank-window expansion: starting
        from the top-``num_common_tokens`` of each model, widen the window by
        one until at least ``num_common_tokens`` tokens are common to both
        rankings (or the candidate pool is exhausted).
        """
        cfg = self.config
        vocab = base_logp.shape[-1]
        target_n = max(1, int(cfg.num_common_tokens))

        # Descending rank of every token for both models.
        base_rank = torch.argsort(base_logp, dim=-1, descending=True)
        expert_rank = torch.argsort(expert_logp, dim=-1, descending=True)

        # Cap the search pool at top_k (behaviour-preserving optimisation).
        pool = vocab
        if 0 < cfg.top_k < vocab:
            pool = max(cfg.top_k, target_n)
        base_rank = base_rank[:, :pool]
        expert_rank = expert_rank[:, :pool]

        out = torch.zeros_like(base_logp, dtype=torch.bool)
        for row in range(base_logp.shape[0]):
            b_list = base_rank[row].tolist()
            e_list = expert_rank[row].tolist()
            common: set = set()
            iter_range = target_n
            limit = min(len(b_list), len(e_list))
            while len(common) < target_n:
                cur = set(b_list[:iter_range]) & set(e_list[:iter_range])
                common.update(cur)
                iter_range += 1
                if iter_range > limit:
                    break
            if not common:
                # Degenerate: no overlap at all -> fall back to base top-1.
                common = {b_list[0]}
            idx = torch.tensor(sorted(common), device=base_logp.device, dtype=torch.long)
            out[row, idx] = True
        return out

    def combine(
        self,
        scores: torch.Tensor,
        guide_scores: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        step = max(0, input_ids.shape[1] - self.prompt_length)

        target = scores.float()
        guide = guide_scores.float()

        # Constant-alpha-for-m-steps-then-off schedule: outside the window
        # SafeDecoding is switched off and the target decodes normally.
        if step >= cfg.window() or cfg.alpha == 0.0:
            return self._apply_banned(scores)

        if cfg.clip_negative_inf:
            guide = torch.where(torch.isinf(guide), target, guide)

        # Authors operate on log_softmax probabilities of each model.
        base_logp = torch.log_softmax(target, dim=-1)
        expert_logp = torch.log_softmax(guide, dim=-1)

        # Step 1: sample space = iterative common-token intersection.
        common_mask = self._common_token_mask(base_logp, expert_logp)

        # Step 2: steered probability P_base + alpha * (P_expert - P_base).
        base_p = base_logp.exp()
        expert_p = expert_logp.exp()
        blended = base_p + float(cfg.alpha) * (expert_p - base_p)
        blended = blended.clamp(min=1e-8)

        # Restrict to the common set and renormalise (softmax over log-probs
        # of the common tokens only, matching the authors' final softmax).
        neg_inf = torch.full_like(blended, float("-inf"))
        common_logits = torch.where(common_mask, blended.log(), neg_inf)
        renorm = torch.log_softmax(common_logits, dim=-1)
        return self._apply_banned(renorm.to(scores.dtype))

    def _apply_banned(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask ``banned_tokens`` to ``-inf`` (never emitted). No-op if unset."""
        cfg = self.config
        if not cfg.banned_tokens:
            return logits
        logits = logits.clone()
        idx = torch.as_tensor(cfg.banned_tokens, device=logits.device, dtype=torch.long)
        logits[..., idx] = float("-inf")
        return logits


__all__ = ["SafeDecodingProcessor", "SafeDecodingConfig"]
