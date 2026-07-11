"""Shared helpers for refusal-direction and steering modules.

Kept at the top level to avoid creating a sibling import cycle between
``safetune.steer`` and ``safetune.evaluate``.
"""
from __future__ import annotations

from typing import Any, List


def _get_decoder_layers(model: Any) -> List[Any]:
    """Return the list of decoder blocks for a causal LM, or [].

    Order of checks mirrors the HF model zoo:

    * Gemma-3: ``model.model.language_model.layers``
    * Llama / Mistral / Qwen / Gemma: ``model.model.layers``
    * GPT-2 / GPT-NeoX: ``model.transformer.h``
    * Falcon: ``model.transformer.h`` (covered by the GPT path)
    * Already-unwrapped inner model: ``model.layers``
    """
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return list(model.model.language_model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "layers"):
        return list(model.layers)
    return []


__all__ = ["_get_decoder_layers"]
