"""
CAA: Contrastive Activation Addition (Panickssery et al., arXiv:2312.06681).

CAA is the foundational additive-steering baseline. Given contrast pairs
``(positive, negative)``, it computes a behavior vector

    v_layer = mean(activations[positive], layer) - mean(activations[negative], layer)

and applies it at inference by adding ``alpha * v_layer`` to the residual
stream of a chosen set of layers. The "behavior" axis is whatever the
contrast pairs encode: refusal vs. compliance, helpfulness vs. dismissal,
honesty vs. deception, etc.

Differences from ``refusal_direction``:

* CAA is behavior-agnostic. Pass any prompt-pair set; the math is the same.
* CAA steers across many layers simultaneously rather than picking one.
* CAA applies *additive* steering only. Use ``refusal_direction`` if you
  also need projective ablation.

Reference paper: N. Panickssery, N. Gabrieli, J. Schulz, M. Tong, E. Hubinger,
A. M. Turner. "Steering Llama 2 via Contrastive Activation Addition."
arXiv:2312.06681 (2023, refreshed 2024).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from safetune._refusal_helpers import _get_decoder_layers

logger = logging.getLogger(__name__)


@dataclass
class CAAConfig:
    """Configuration for CAA vector extraction and application.

    Attributes:
        target_layers: Layers to extract / apply at. Defaults to all decoder
            layers if ``None``.
        pool_method: Sequence-dim reduction at extraction time.
            ``"last_token"`` is paper-faithful; ``"mean"`` averages.
        strength: Multiplier ``alpha`` applied at inference time.
        normalize: Whether to L2-normalize each per-layer vector.
    """

    target_layers: Optional[List[int]] = None
    pool_method: str = "last_token"
    strength: float = 1.0
    normalize: bool = False


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_caa_vectors(
    model: nn.Module,
    tokenizer: Any,
    positive_prompts: List[str],
    negative_prompts: List[str],
    config: Optional[CAAConfig] = None,
) -> Dict[int, torch.Tensor]:
    """Extract per-layer CAA steering vectors from contrast pairs.

    Returns a dict ``{layer_idx: 1-D tensor (hidden,)}``.
    """
    cfg = config or CAAConfig()

    layers = _get_decoder_layers(model)
    if not layers:
        raise RuntimeError(
            "extract_caa_vectors: could not locate decoder layers. "
            "Expected model.model.language_model.layers (Gemma-3), "
            "model.model.layers (Llama/Mistral/Qwen/Gemma), or "
            "model.transformer.h (GPT-style)."
        )

    from safetune.core.runtime.inference.vector_extraction import (
        SteeringVectorExtractor,
        VectorExtractionConfig,
    )

    target = cfg.target_layers if cfg.target_layers is not None else list(range(len(layers)))
    ve_cfg = VectorExtractionConfig(
        target_layers=target,
        pool_method=cfg.pool_method,
        normalize=cfg.normalize,
    )
    ext = SteeringVectorExtractor(model, tokenizer, ve_cfg)
    # Extractor returns ``safe_mean - unsafe_mean``; the CAA convention is
    # ``positive - negative``, so map positive -> safe_prompts.
    vectors = ext.extract(safe_prompts=positive_prompts, unsafe_prompts=negative_prompts)
    logger.info("CAA: extracted %d per-layer behavior vectors.", len(vectors))
    return vectors


# ---------------------------------------------------------------------------
# Runtime intervention
# ---------------------------------------------------------------------------

class CAAModel:
    """Apply CAA steering at inference time.

    Example::

        vecs = extract_caa_vectors(model, tok, positive, negative, cfg)
        with CAAModel(model, vecs, strength=1.5) as steered:
            outputs = model.generate(**inputs)
    """

    def __init__(
        self,
        model: nn.Module,
        vectors: Dict[int, torch.Tensor],
        strength: float = 1.0,
    ) -> None:
        self.model = model
        self.vectors = {int(k): v.detach().clone() for k, v in vectors.items()}
        self.strength = float(strength)
        self._handles: List[Any] = []
        self.install()

    def _make_hook(self, vec: torch.Tensor):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            v = vec.to(dtype=h.dtype, device=h.device)
            h = h + self.strength * v
            if is_tuple:
                return (h,) + output[1:]
            return h
        return hook

    def install(self) -> "CAAModel":
        self.remove()
        layers = _get_decoder_layers(self.model)
        for idx, vec in self.vectors.items():
            if 0 <= idx < len(layers):
                self._handles.append(layers[idx].register_forward_hook(self._make_hook(vec)))
        logger.info("CAAModel: installed %d hooks (strength=%.2f).", len(self._handles), self.strength)
        return self

    def remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        return getattr(self.__dict__["model"], name)

    def __enter__(self) -> "CAAModel":
        return self.install()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()


__all__ = [
    "CAAConfig",
    "CAAModel",
    "extract_caa_vectors",
]
