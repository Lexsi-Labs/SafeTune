"""HuggingFace ``transformers`` inference backend.

Slow but always available. Used as the default fallback when vLLM is not
installed or the host has no GPU.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

import torch
import torch.nn as nn

from .base import GenerationConfig, InferenceBackend

logger = logging.getLogger(__name__)


class TransformersBackend(InferenceBackend):
    """Inference via ``AutoModelForCausalLM.generate``.

    The model may be a HuggingFace hub id, a local path, or a pre-instantiated
    ``nn.Module``. In the last case ``tokenizer`` must be supplied.

    Args:
        model: hub id, local path, or ``nn.Module``.
        tokenizer: required if ``model`` is an ``nn.Module``; otherwise
            inferred from ``model`` via ``AutoTokenizer.from_pretrained``.
        device: ``"cuda"``, ``"cuda:1"``, ``"cpu"``, or ``"auto"`` (default).
        torch_dtype: ``torch.bfloat16`` by default on CUDA, ``torch.float32`` on CPU.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        config: Optional[GenerationConfig] = None,
        chat_template: bool = True,
        system_prompt: Optional[str] = None,
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__(
            model=model if isinstance(model, str) else getattr(model, "name_or_path", "in-memory"),
            config=config,
            chat_template=chat_template,
            system_prompt=system_prompt,
        )
        self._device = device
        self._dtype = torch_dtype
        self._trust_remote = trust_remote_code

        if isinstance(model, nn.Module):
            if tokenizer is None:
                raise ValueError(
                    "TransformersBackend: when ``model`` is an nn.Module, "
                    "``tokenizer`` must be provided."
                )
            self._loaded_model: nn.Module = model
            self._loaded_tokenizer = tokenizer
        else:
            self._loaded_model = None  # type: ignore[assignment]
            self._loaded_tokenizer = tokenizer
            self._loaded_model_name: str = model

    # ------------------------------------------------------------------ load

    def _ensure_loaded(self) -> None:
        if self._loaded_model is not None and self._loaded_tokenizer is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._loaded_tokenizer is None:
            self._loaded_tokenizer = AutoTokenizer.from_pretrained(
                self._loaded_model_name,
                trust_remote_code=self._trust_remote,
            )
            if self._loaded_tokenizer.pad_token is None:
                self._loaded_tokenizer.pad_token = self._loaded_tokenizer.eos_token
            self._loaded_tokenizer.padding_side = "left"

        if self._loaded_model is None:
            dtype = self._dtype
            if dtype is None:
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device_map = self._device if self._device != "auto" else "auto"
            logger.info(
                "TransformersBackend: loading %s (dtype=%s, device=%s)",
                self._loaded_model_name,
                dtype,
                device_map,
            )
            self._loaded_model = AutoModelForCausalLM.from_pretrained(
                self._loaded_model_name,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=self._trust_remote,
            )
            self._loaded_model.eval()

    # --------------------------------------------------------------- generate

    def generate(self, prompts: List[str]) -> List[str]:
        self._ensure_loaded()
        tok = self._loaded_tokenizer
        model = self._loaded_model
        cfg = self.config

        if self.chat_template:
            try:
                formatted = [self._apply_chat_template(tok, p) for p in prompts]
            except Exception as e:
                logger.warning(
                    "TransformersBackend: chat_template failed (%s); using raw prompts.", e
                )
                formatted = list(prompts)
        else:
            formatted = list(prompts)

        device = next(model.parameters()).device
        responses: List[str] = []
        bs = max(1, cfg.batch_size)
        for i in range(0, len(formatted), bs):
            batch = formatted[i : i + bs]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=cfg.temperature > 0,
                    pad_token_id=tok.pad_token_id,
                )
                if cfg.temperature > 0:
                    gen_kwargs["temperature"] = cfg.temperature
                    gen_kwargs["top_p"] = cfg.top_p
                if cfg.repetition_penalty != 1.0:
                    gen_kwargs["repetition_penalty"] = cfg.repetition_penalty
                if cfg.seed is not None:
                    torch.manual_seed(cfg.seed)
                out = model.generate(**enc, **gen_kwargs)
            input_len = enc["input_ids"].shape[1]
            for ids in out:
                new = ids[input_len:]
                text = tok.decode(new, skip_special_tokens=True)
                responses.append(self._truncate_on_stop(text))
        return responses
