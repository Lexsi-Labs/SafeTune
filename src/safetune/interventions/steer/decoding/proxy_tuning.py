"""
Proxy Tuning (Liu et al., 2024, arXiv:2401.08565).

Logit arithmetic across a small proxy pair to shift a large untuned model:

    final = target_large + (proxy_tuned_small - proxy_base_small)

For safety we instantiate this as:

* ``target_large``: a large unaligned base model
* ``proxy_tuned_small``: a small aligned instruct model
* ``proxy_base_small``: the small pre-aligned base

The delta ``proxy_tuned - proxy_base`` represents "what alignment does"; we
add it to the large model's logits to bring in safety behaviour without
touching the large model's weights. Original paper closes 88% of the Llama-2
70B -> 70B-Chat gap with 7B proxies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from ._base import GuidedLogitsProcessor


@dataclass
class ProxyTuningConfig:
    """Configuration for Proxy Tuning.

    Attributes:
        scale: multiplier on the proxy delta. 1.0 is the paper default.
        clamp_delta: optional cap on |delta| at each token to avoid runaway
            logit shifts. ``None`` disables.
    """

    scale: float = 1.0
    clamp_delta: Optional[float] = None


class ProxyTuningProcessor(GuidedLogitsProcessor):
    """``LogitsProcessor`` implementing Proxy Tuning.

    Note: a single :class:`GuidedLogitsProcessor` runs one auxiliary model.
    Proxy Tuning needs TWO. We compose: pass ``proxy_tuned`` as ``guide`` and
    ``proxy_base`` as ``base_guide``. The processor maintains both forward
    passes and computes ``guide - base_guide`` as the delta.
    """

    def __init__(
        self,
        proxy_tuned: Any,
        proxy_base: Any,
        tokenizer_target: Any,
        tokenizer_proxy: Optional[Any] = None,
        config: Optional[ProxyTuningConfig] = None,
    ) -> None:
        super().__init__(guide=proxy_tuned, tokenizer_target=tokenizer_target,
                         tokenizer_guide=tokenizer_proxy)
        self.proxy_base = proxy_base
        self.config = config or ProxyTuningConfig()

    @torch.no_grad()
    def _proxy_base_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.proxy_base(input_ids=input_ids, use_cache=False)
        native = out.logits[:, -1, :].float()
        if self._target_to_guide.numel() == 0:
            return native
        translation = self._target_to_guide.to(native.device)
        mask = translation >= 0
        gathered = native[:, translation.clamp(min=0)]
        return torch.where(mask, gathered, torch.zeros_like(gathered))

    def combine(
        self,
        scores: torch.Tensor,
        guide_scores: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        tuned = torch.where(torch.isinf(guide_scores), torch.zeros_like(guide_scores), guide_scores)
        base = self._proxy_base_logits(input_ids)
        delta = self.config.scale * (tuned - base)
        if cfg.clamp_delta is not None:
            delta = delta.clamp(-float(cfg.clamp_delta), float(cfg.clamp_delta))
        return (scores.float() + delta).to(scores.dtype)


__all__ = ["ProxyTuningProcessor", "ProxyTuningConfig"]
