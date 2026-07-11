"""vLLM offline batched inference backend.

Fastest single-GPU option. Falls back to ``TransformersBackend`` when vLLM
is not installed.

Mirrors the cthetha-eval vLLM path: chat-template wraps prompts before
tokenization, then a single ``llm.generate`` call drives the whole batch.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from .base import GenerationConfig, InferenceBackend

logger = logging.getLogger(__name__)


class VllmBackend(InferenceBackend):
    """Run inference through vLLM's offline ``LLM.generate``.

    Args:
        model: HF hub id or local path to a model directory.
        config: shared :class:`GenerationConfig`.
        tensor_parallel_size: vLLM TP. Default 1.
        dtype: vLLM dtype (``"bfloat16"`` recommended).
        gpu_memory_utilization: vLLM memory fraction (default 0.9).
        max_model_len: vLLM context window. Default 4096.
        enforce_eager: bypass CUDA graphs. Slower but more compatible.
        attention_backend: optional override (e.g. ``"TRITON_ATTN"`` for
            Blackwell GPUs where FlashAttn PTX still has issues).
    """

    def __init__(
        self,
        model: str,
        config: Optional[GenerationConfig] = None,
        chat_template: bool = True,
        system_prompt: Optional[str] = None,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        enforce_eager: bool = True,
        attention_backend: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            config=config,
            chat_template=chat_template,
            system_prompt=system_prompt,
        )
        self._tp = tensor_parallel_size
        self._dtype = dtype
        self._gpu_mem = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._enforce_eager = enforce_eager
        self._attn_backend = attention_backend
        self._trust_remote = trust_remote_code
        self._loaded_llm: Any = None
        self._loaded_tokenizer: Any = None
        self._loaded_sampling: Any = None

    @staticmethod
    def is_available() -> bool:
        try:
            import vllm  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        if self._loaded_llm is not None:
            return
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        logger.info("VllmBackend: loading %s (tp=%d, dtype=%s)", self.model, self._tp, self._dtype)
        kwargs = dict(
            model=self.model,
            dtype=self._dtype,
            tensor_parallel_size=self._tp,
            max_model_len=self._max_model_len,
            gpu_memory_utilization=self._gpu_mem,
            trust_remote_code=self._trust_remote,
            enforce_eager=self._enforce_eager,
        )
        if self._attn_backend:
            try:
                from vllm.v1.attention.backends.registry import AttentionBackendEnum
                kwargs["attention_backend"] = getattr(AttentionBackendEnum, self._attn_backend)
            except Exception as e:
                logger.warning("VllmBackend: %s unavailable (%s); using vLLM default.",
                               self._attn_backend, e)

        self._loaded_llm = LLM(**kwargs)
        self._loaded_tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=self._trust_remote
        )
        if self._loaded_tokenizer.pad_token is None:
            self._loaded_tokenizer.pad_token = self._loaded_tokenizer.eos_token
        cfg = self.config
        self._loaded_sampling = SamplingParams(
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p if cfg.temperature > 0 else 1.0,
            repetition_penalty=cfg.repetition_penalty,
            stop=cfg.stop_sequences or None,
            seed=cfg.seed,
        )

    def generate(self, prompts: List[str]) -> List[str]:
        self._ensure_loaded()
        if self.chat_template:
            try:
                formatted = [self._apply_chat_template(self._loaded_tokenizer, p) for p in prompts]
            except Exception as e:
                logger.warning("VllmBackend: chat template failed (%s); using raw prompts.", e)
                formatted = list(prompts)
        else:
            formatted = list(prompts)
        outputs = self._loaded_llm.generate(formatted, self._loaded_sampling)
        return [self._truncate_on_stop(o.outputs[0].text) for o in outputs]
