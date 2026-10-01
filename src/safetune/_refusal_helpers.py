"""Shared helpers for refusal-direction and steering modules, and for loading
and walking HF models (text-only and vision-language).

Kept at the top level to avoid creating a sibling import cycle between
``safetune.steer`` and ``safetune.evaluate``.
"""
from __future__ import annotations

import contextlib
from typing import Any, Iterator, List, Optional


def _get_decoder_layers(model: Any) -> List[Any]:
    """Return the list of decoder blocks for a causal LM, or [].

    Order of checks mirrors the HF model zoo:

    * Vision-language (Gemma-3, Aya Vision, North): ``model.model.language_model.layers``
    * Llama / Mistral / Qwen / Gemma / Cohere: ``model.model.layers``
    * Inner vision-language model: ``model.language_model.layers``
    * GPT-2 / Falcon: ``model.transformer.h``
    * GPT-NeoX / Pythia: ``model.gpt_neox.layers``
    * Already-unwrapped inner model: ``model.layers``

    A peft model is unwrapped first.
    """
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return list(model.model.language_model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return list(model.language_model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    return []


_VISION_TOWER_NAMES = ("vision_tower", "visual", "vision_model")


def _layer_index(name: str) -> Optional[int]:
    """Decoder-layer index in a parameter or module name (``...layers.<N>...``),
    or None. Vision-tower blocks (Aya Vision's ``vision_tower.encoder.layers.<N>``)
    are not decoder layers and give None."""
    parts = name.split(".")
    if any(v in parts for v in _VISION_TOWER_NAMES):
        return None
    for i, p in enumerate(parts[:-1]):
        if p == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def _is_vision_config(config: Any) -> bool:
    """True for a vision-language config that ``AutoModelForCausalLM`` cannot load
    (Aya Vision, North). Configs in both mappings (Gemma-3) stay causal LMs."""
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES)
    mt = getattr(config, "model_type", None)
    return mt not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES and mt in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES


def _auto_model_class(config: Any):
    """``AutoModelForImageTextToText`` for vision-language configs, else
    ``AutoModelForCausalLM``."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    return AutoModelForImageTextToText if _is_vision_config(config) else AutoModelForCausalLM


def _load_pretrained_lm(path: str, *, trust_remote_code: bool = False, **kwargs: Any):
    """``from_pretrained`` with the auto class that fits ``path``'s config; ``kwargs``
    go to ``from_pretrained`` unchanged."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
    fusion = getattr(config, "fusion_config", None)
    if fusion and "fusion_config" not in kwargs:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        if get_checkpoint_conversion_mapping(config.model_type):
            # ponytail: transformers 5.17 registers a fused model's (North's) checkpoint
            # converters process-wide on the first from_pretrained and raises
            # "conflicts with an existing conversion mapping" on the second. They are
            # already registered, so skip that step; the model is still fused.
            # Drop this once transformers makes the registration idempotent.
            kwargs["fusion_config"] = {}
    model = _auto_model_class(config).from_pretrained(
        path, trust_remote_code=trust_remote_code, **kwargs)
    if fusion and kwargs.get("fusion_config") == {}:
        model.config.fusion_config = fusion  # saved checkpoints keep it
    return model


def _lora_target_modules(model: Any, modules: Any) -> Any:
    """LoRA ``target_modules`` for ``model``: ``modules`` unchanged, except that on a
    vision-language model a list of names becomes a peft regex scoped to the
    language model, so LoRA (and the merged save) leave the vision tower alone.
    A regex string or None passes through."""
    if modules is None or isinstance(modules, str) or not _is_vision_config(model.config):
        return modules
    import re
    return rf".*language_model\..*\.({'|'.join(re.escape(m) for m in modules)})"


def _encode_prompts(tokenizer: Any, prompts: List[str], chat_template: bool = True, **kwargs: Any):
    """Tokenize ``prompts`` the way SafeTune's generators do (``evaluate()``,
    ``TransformersBackend``, steering-vector extraction): each prompt as one user
    turn through the chat template, with no extra special tokens, when
    ``chat_template`` is set and the tokenizer has one; else as raw text.

    Calibration that reads hidden states must use this so it sees the same
    tokens as generation."""
    if chat_template and getattr(tokenizer, "chat_template", None):
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                                 tokenize=False, add_generation_prompt=True)
                   for p in prompts]
        kwargs["add_special_tokens"] = False
    return tokenizer(prompts, **kwargs)


@contextlib.contextmanager
def _left_padding(tokenizer: Any) -> Iterator[Any]:
    """Left-pad ``tokenizer`` inside the block and restore its padding side after.

    Batched generation and last-token reads need left padding: with right
    padding the pads sit between a short prompt and its first new token.
    Training tokenizers are right-padded (``load_tok``), so every batched
    generation path wraps its tokenize + generate in this."""
    prev = getattr(tokenizer, "padding_side", None)
    if prev is not None:
        tokenizer.padding_side = "left"
    try:
        yield tokenizer
    finally:
        if prev is not None:
            tokenizer.padding_side = prev


def _strip_bos(tokenizer: Any, texts: List[str]) -> List[str]:
    """Drop the leading BOS text from chat-templated ``texts`` when ``tokenizer``
    adds BOS itself. vLLM tokenizes a text prompt with special tokens, so a
    template that already renders BOS would otherwise start with two."""
    bos = getattr(tokenizer, "bos_token", None)
    if not bos or not any(t.startswith(bos) for t in texts):
        return list(texts)
    if tokenizer.encode("a")[:1] != [tokenizer.bos_token_id]:
        return list(texts)  # tokenizer adds no BOS: the template's is the only one
    return [t[len(bos):] if t.startswith(bos) else t for t in texts]


__all__ = ["_get_decoder_layers", "_encode_prompts", "_strip_bos", "_left_padding", "_layer_index", "_is_vision_config", "_auto_model_class",
           "_load_pretrained_lm", "_lora_target_modules"]
