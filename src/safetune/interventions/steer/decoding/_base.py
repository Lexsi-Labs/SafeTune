"""
Shared base for logit-level steering processors.

``GuidedLogitsProcessor`` runs a second model in parallel and exposes its
next-token logits to subclasses for blending. We keep the past-key-values
to avoid recomputing the guide's prefix on every step.

For HF's ``model.generate(..., logits_processor=LogitsProcessorList(...))``
contract, the processor's ``__call__`` signature is
``(input_ids, scores) -> scores``. The first call sees the full prompt
prefix in ``input_ids`` and an initial ``scores`` matrix of shape
``(batch, vocab)``. Subsequent calls extend ``input_ids`` by one column.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

try:
    from transformers import LogitsProcessor
except ImportError:  # pragma: no cover
    class LogitsProcessor:  # type: ignore[no-redef]
        def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
            return scores


logger = logging.getLogger(__name__)


@dataclass
class _GuideState:
    """Per-batch cache for a guide model's KV state."""

    past_key_values: Any = None
    last_seen_len: int = 0
    batch_size: int = 0


class GuidedLogitsProcessor(LogitsProcessor):
    """Base class: keeps a parallel forward pass on a guide model.

    Subclasses override :meth:`combine(scores, guide_scores, input_ids)` to
    produce the final logits.

    Args:
        guide: an ``nn.Module`` answering ``forward(input_ids, ...)`` with a
            ``.logits`` attribute, e.g. an HF causal LM in eval mode.
        tokenizer_target: tokenizer of the target model (vocab id space).
        tokenizer_guide: tokenizer of the guide model. If different from
            ``tokenizer_target``, only the intersection of the two vocabs is
            used and the guide contribution is zero elsewhere.
    """

    def __init__(
        self,
        guide: Any,
        tokenizer_target: Any,
        tokenizer_guide: Optional[Any] = None,
    ) -> None:
        self.guide = guide
        self.tokenizer_target = tokenizer_target
        self.tokenizer_guide = tokenizer_guide or tokenizer_target
        self._state = _GuideState()
        # Pre-compute the (target_id -> guide_id) translation table.
        # When the vocabs are identical, this is the identity; otherwise we
        # map by token string and accept lossy mismatches (mapped to -1).
        self._target_to_guide = self._build_translation()

    def _build_translation(self) -> torch.Tensor:
        tv = getattr(self.tokenizer_target, "get_vocab", lambda: {})()
        gv = getattr(self.tokenizer_guide, "get_vocab", lambda: {})()
        if not tv or not gv:
            return torch.empty(0, dtype=torch.long)
        if tv == gv:
            # Identical vocab: use the guide's native logits directly. (Do NOT
            # build an arange over len(tokenizer) — modern lm_heads are padded
            # WIDER than the tokenizer vocab, e.g. Qwen2.5 head 151936 vs vocab
            # 151665, and gathering over len(vocab) would truncate the logits to
            # the wrong width and crash combine().)
            return torch.empty(0, dtype=torch.long)
        out = torch.full((len(tv),), -1, dtype=torch.long)
        for tok_str, tid in tv.items():
            gid = gv.get(tok_str)
            if gid is not None:
                out[tid] = gid
        return out

    def _attention_mask(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Derive an attention mask so a left-padded *batch* forwards correctly.

        For a single (unpadded) prompt this is all-ones and a no-op. For a
        batched generation call the prompts are left-padded, so the guide must
        not attend to the leading pad tokens. The mask marks every position
        from the first non-pad token onward as real (``cummax`` keeps a
        mid-sequence pad/eos token attended once real content has started).
        """
        pad_id = getattr(self.tokenizer_target, "pad_token_id", None)
        if pad_id is None:
            return None
        real = (input_ids != pad_id)
        return real.cummax(dim=1).values.long()

    @torch.no_grad()
    def _guide_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run guide on the same input_ids; return last-step logits in target-vocab space.

        Uses the guide's KV cache: the first call forwards the whole prompt;
        each later call forwards only the tokens added since the previous call
        (``_GuideState.last_seen_len``), reusing the cached past. This turns the
        per-step guide cost from O(sequence) to O(1) — a ~100x speedup over a
        full recompute, with identical last-step logits. The processor is
        rebuilt per generation batch (see the steer runner), so ``_state`` is
        always fresh for a batch and there is no cross-batch cache staleness.
        Skipped steps (a subclass that returns before calling the guide) are
        handled naturally — ``last_seen_len`` only advances on an actual call,
        so the next call simply forwards the accumulated new tokens.
        """
        # Re-tokenize is impossible without raw text; we use the same input_ids
        # provided the tokenizers share the vocab.
        attn = self._attention_mask(input_ids)
        st = self._state
        seen = st.last_seen_len
        total = input_ids.shape[1]
        # Incremental decode is valid only when the cache holds a strict prefix
        # of the current sequence for the same batch size; otherwise recompute.
        cache_ok = (
            st.past_key_values is not None
            and 0 < seen < total
            and input_ids.shape[0] == st.batch_size
        )
        if cache_ok:
            out = self.guide(input_ids=input_ids[:, seen:], attention_mask=attn,
                             past_key_values=st.past_key_values, use_cache=True)
        else:
            # first call for this batch (or a reset) — full prompt forward
            out = self.guide(input_ids=input_ids, attention_mask=attn,
                             use_cache=True)
        st.past_key_values = getattr(out, "past_key_values", None)
        st.last_seen_len = total
        st.batch_size = input_ids.shape[0]
        guide_scores_native = out.logits[:, -1, :].float()
        translation = self._target_to_guide
        if translation.numel() == 0:
            # No vocab info; return the native logits assuming aligned space.
            return guide_scores_native
        # Build target-vocab-shaped scores by gathering from guide.
        translation = translation.to(guide_scores_native.device)
        mask = translation >= 0
        idx_safe = translation.clamp(min=0)
        gathered = guide_scores_native[:, idx_safe]
        # For target tokens with no guide counterpart, set logits to -inf so
        # the guide contributes nothing. The subclass can choose to ignore.
        neg_inf = torch.full_like(gathered, float("-inf"))
        return torch.where(mask, gathered, neg_inf)

    # Subclasses override this.
    def combine(
        self,
        scores: torch.Tensor,
        guide_scores: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the guide model with self as a logits_processor."""
        from transformers import LogitsProcessorList
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            input_ids = args[0] if args else None
        if input_ids is not None and hasattr(self, "prompt_length"):
            self.prompt_length = input_ids.shape[1]
        self._state = _GuideState()
        lp = kwargs.pop("logits_processor", None)
        if lp is None:
            lp = LogitsProcessorList([self])
        else:
            lp = LogitsProcessorList([self] + list(lp))
        return self.guide.generate(*args, logits_processor=lp, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "guide":
            raise AttributeError(name)
        return getattr(self.__dict__["guide"], name)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        guide_scores = self._guide_logits(input_ids)
        # Reconcile width to the target model's logit width. An lm_head is often
        # padded WIDER than the tokenizer vocab (e.g. Qwen2.5: head 151936 vs
        # vocab 151665), and a cross-vocab guide maps to len(target_vocab); in
        # both cases guide_scores can be narrower/wider than `scores`. Pad the
        # tail with -inf (guide abstains on those ids) or truncate to match, so
        # subclass combine() never hits a shape mismatch.
        gw, sw = guide_scores.shape[-1], scores.shape[-1]
        if gw < sw:
            pad = torch.full(
                (*guide_scores.shape[:-1], sw - gw),
                float("-inf"), dtype=guide_scores.dtype, device=guide_scores.device,
            )
            guide_scores = torch.cat([guide_scores, pad], dim=-1)
        elif gw > sw:
            guide_scores = guide_scores[..., :sw]
        return self.combine(scores, guide_scores, input_ids)


__all__ = ["GuidedLogitsProcessor", "LogitsProcessor"]
