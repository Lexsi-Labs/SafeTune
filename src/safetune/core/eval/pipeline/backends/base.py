"""
Inference backend abstraction.

A backend takes a list of chat-formatted prompts and returns a list of model
completions. Concrete subclasses wrap different runtimes:

* ``TransformersBackend``: HuggingFace ``AutoModelForCausalLM.generate``.
  Default fallback. Slow but always available.
* ``VllmBackend``: vLLM offline batched inference. Fast on a single GPU.
* ``ApiBackend``: any OpenAI-compatible HTTP endpoint (OpenAI, Anthropic,
  Together, vLLM's OpenAI server, a SageMaker endpoint with the right shim).
  Useful when you do not have GPU on the host but do have credit.
* ``DryRunBackend``: returns canned strings, no model load. Used in tests.

The same Generator can swap backends with a one-line config change. This
mirrors the cthetha-eval design but as a Python class hierarchy rather than
a CLI script.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenerationConfig:
    """Sampling parameters shared across all backends.

    Attributes:
        max_new_tokens: Cap on generated tokens (excluding the prompt).
        temperature: Sampling temperature. ``0.0`` requests greedy decoding.
        top_p: Nucleus sampling threshold. Ignored when temperature == 0.
        repetition_penalty: HF-style repetition penalty. ``1.0`` disables it.
        stop_sequences: Optional list of stop strings. Backends that do not
            support stop tokens natively truncate post-hoc.
        seed: RNG seed when supported (most backends respect this).
        batch_size: Number of prompts per forward pass. Backends may ignore
            this if they handle batching internally (e.g. vLLM).
    """

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop_sequences: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    batch_size: int = 8


class InferenceBackend(abc.ABC):
    """Abstract base for inference backends.

    Subclasses implement ``generate(prompts)`` returning one string per input
    prompt. The base class handles chat-template wrapping and stop-string
    truncation so subclass implementations stay small.
    """

    def __init__(
        self,
        model: str,
        config: Optional[GenerationConfig] = None,
        chat_template: bool = True,
        system_prompt: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.config = config or GenerationConfig()
        self.chat_template = chat_template
        self.system_prompt = system_prompt
        self.extra = extra or {}

    # ----------------------------------------------------------------- API

    @abc.abstractmethod
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate one completion per prompt, in input order."""

    def __call__(self, prompts: List[str]) -> List[str]:
        return self.generate(prompts)

    # ---------------------------------------------------- helper utilities

    def _apply_chat_template(self, tokenizer: Any, prompt: str) -> str:
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _truncate_on_stop(self, text: str) -> str:
        """Cut ``text`` at the first occurrence of any stop sequence."""
        if not self.config.stop_sequences:
            return text
        idx = len(text)
        for s in self.config.stop_sequences:
            j = text.find(s)
            if 0 <= j < idx:
                idx = j
        return text[:idx]
