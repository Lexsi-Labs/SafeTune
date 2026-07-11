"""Inference backend registry.

Use :func:`make_backend` for a one-line dispatch that picks a backend by name
and gracefully degrades when an optional dependency (vLLM) is missing.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .base import GenerationConfig, InferenceBackend
from .dryrun import DryRunBackend
from .transformers import TransformersBackend

logger = logging.getLogger(__name__)


def make_backend(
    backend: str,
    model: Any,
    config: Optional[GenerationConfig] = None,
    **kwargs: Any,
) -> InferenceBackend:
    """Instantiate a backend by name.

    Supported names:
      ``"transformers"`` / ``"hf"``  : :class:`TransformersBackend`
      ``"vllm"``                     : :class:`VllmBackend` (falls back to HF if vLLM missing)
      ``"vllm-lens"`` / ``"vllm_lens"`` : :class:`VllmSteeredBackend`
        (requires ``steering_vectors`` in ``kwargs``; falls back to plain vLLM if
        ``vllm-lens`` is missing AND ``steering_vectors`` is empty)
      ``"api"``                      : :class:`ApiBackend`
      ``"dryrun"``                   : :class:`DryRunBackend`

    ``kwargs`` is forwarded to the backend constructor unchanged.
    """
    name = backend.lower()
    if name in ("transformers", "hf", "huggingface"):
        return TransformersBackend(model=model, config=config, **kwargs)
    if name == "vllm":
        from .vllm import VllmBackend
        if not VllmBackend.is_available():
            logger.warning(
                "make_backend: vLLM requested but not installed; falling back to TransformersBackend."
            )
            return TransformersBackend(model=model, config=config, **kwargs)
        return VllmBackend(model=model, config=config, **kwargs)
    if name in ("vllm-lens", "vllm_lens"):
        from .vllm_lens import VllmSteeredBackend
        if not VllmSteeredBackend.is_available():
            logger.warning(
                "make_backend: vllm-lens requested but not installed. "
                "Install with `pip install vllm vllm-lens`."
            )
            # Without vllm-lens we cannot steer; fall back to plain vLLM if
            # available, else HF. The Generator will run inference without
            # the requested intervention; warn the user upstream.
            from .vllm import VllmBackend
            if VllmBackend.is_available():
                kwargs.pop("steering_vectors", None)
                return VllmBackend(model=model, config=config, **kwargs)
            kwargs.pop("steering_vectors", None)
            return TransformersBackend(model=model, config=config, **kwargs)
        return VllmSteeredBackend(model=model, config=config, **kwargs)
    if name == "api":
        from .api import ApiBackend
        return ApiBackend(model=model, config=config, **kwargs)
    if name == "dryrun":
        return DryRunBackend(model=str(model), config=config, **kwargs)
    raise ValueError(f"Unknown backend: {backend!r}")


__all__ = [
    "GenerationConfig",
    "InferenceBackend",
    "TransformersBackend",
    "DryRunBackend",
    "make_backend",
]
