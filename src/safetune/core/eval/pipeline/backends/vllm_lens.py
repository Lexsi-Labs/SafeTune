"""
vLLM-Lens accelerated steering backend.

Uses `vllm-lens` (UKGovernmentBEIS/vllm-lens) to apply per-layer steering
vectors directly inside vLLM's forward pass, instead of attaching PyTorch
forward hooks on the HF model. The vLLM-Lens plugin auto-registers when
installed (``pip install vllm-lens``) and exposes activation interception
via ``SamplingParams.extra_args``.

Why this matters:
* HF + hooks: roughly 30-50 tokens/sec on Llama-3.2-1B with batch 8.
* vLLM-Lens: 5-10x more tokens/sec at the same batch, with multi-GPU
  tensor / pipeline parallelism free.

This backend slots into the Generator the same way as :class:`VllmBackend`,
but takes one or more refusal-direction / CAA-style vectors at construct
time and applies them on every forward pass. Combine with
:func:`safetune.steer.extract_refusal_direction` to first compute the
vector on a small calibration set (one-time, slow path), then run the full
benchmark at vLLM speed (fast path).

Limitations:
* vLLM-Lens's primary intervention is *additive* steering. Refusal-direction
  *ablation* (project-out) is supported only via the experimental "Phase 2"
  activation_delta path in the upstream RFC. For ablation today, use the
  HF backend with :class:`RefusalDirectionModel`. For steering toward
  refusal (defense), use this backend.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch

from .base import GenerationConfig, InferenceBackend

logger = logging.getLogger(__name__)


@dataclass
class SteeringSpec:
    """One steering vector to apply at a specific set of layers.

    Mirrors :class:`vllm_lens.SteeringVector` but lives in SafeTune so library
    users do not need to import ``vllm_lens`` directly until apply time.

    Attributes:
        vector: 1-D tensor of shape ``(hidden,)``.
        layer_indices: layers at which to apply the vector. Multiple layers
            multiply the effect, so the typical pattern is one or a small
            range (e.g. ``[10]`` or ``[8, 9, 10, 11]``).
        scale: multiplicative strength (``alpha``).
        norm_match: if True, rescale to match the local residual-stream norm
            before adding. Recommended; otherwise the effect varies with
            depth as activations grow.
        position_indices: if set, restrict the steer to those token positions
            (negative indices count from the end of the prompt). ``None``
            applies at every position.
    """

    vector: torch.Tensor
    layer_indices: Sequence[int]
    scale: float = 0.5
    norm_match: bool = True
    position_indices: Optional[Sequence[int]] = None


def _is_available() -> bool:
    try:
        import vllm_lens  # noqa: F401
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


class VllmSteeredBackend(InferenceBackend):
    """vLLM offline inference with always-on activation steering.

    Args:
        model: HF hub id or local path.
        steering_vectors: one or more :class:`SteeringSpec` to apply per call.
        config: shared :class:`GenerationConfig`.
        chat_template / system_prompt: as in the base class.
        tensor_parallel_size / dtype / gpu_memory_utilization / max_model_len:
            forwarded to vLLM's ``LLM`` constructor.

    Example::

        from safetune.steer import extract_refusal_direction, RefusalDirectionConfig
        from safetune.core.eval.pipeline.backends.vllm_lens import (
            VllmSteeredBackend, SteeringSpec,
        )

        # Extract direction on the slow path (HF hooks, one-time).
        # ... direction, picked_layer, _ = extract_refusal_direction(model, tok, ...)

        backend = VllmSteeredBackend(
            model="meta-llama/Llama-3.2-1B-Instruct",
            steering_vectors=[
                SteeringSpec(vector=direction, layer_indices=[picked_layer], scale=0.5),
            ],
        )
        gen = Generator(backend=backend)
        rows = gen.run(prompts)
    """

    def __init__(
        self,
        model: str,
        steering_vectors: List[SteeringSpec],
        config: Optional[GenerationConfig] = None,
        chat_template: bool = True,
        system_prompt: Optional[str] = None,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        enforce_eager: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            config=config,
            chat_template=chat_template,
            system_prompt=system_prompt,
        )
        if not steering_vectors:
            raise ValueError(
                "VllmSteeredBackend: ``steering_vectors`` is empty. "
                "Use VllmBackend (no steering) if you want plain vLLM."
            )
        self.steering_vectors = list(steering_vectors)
        self._tp = tensor_parallel_size
        self._dtype = dtype
        self._gpu_mem = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._enforce_eager = enforce_eager
        self._trust_remote = trust_remote_code
        self._loaded_llm: Any = None
        self._loaded_tokenizer: Any = None

    @staticmethod
    def is_available() -> bool:
        return _is_available()

    # ------------------------------------------------------------------ load

    def _ensure_loaded(self) -> None:
        if self._loaded_llm is not None:
            return
        if not self.is_available():
            raise ImportError(
                "VllmSteeredBackend requires both ``vllm`` and ``vllm-lens``. "
                "Install with: pip install vllm vllm-lens"
            )
        from transformers import AutoTokenizer
        from vllm import LLM

        logger.info(
            "VllmSteeredBackend: loading %s (tp=%d, dtype=%s, vectors=%d)",
            self.model, self._tp, self._dtype, len(self.steering_vectors),
        )
        self._loaded_llm = LLM(
            model=self.model,
            dtype=self._dtype,
            tensor_parallel_size=self._tp,
            max_model_len=self._max_model_len,
            gpu_memory_utilization=self._gpu_mem,
            trust_remote_code=self._trust_remote,
            enforce_eager=self._enforce_eager,
        )
        self._loaded_tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=self._trust_remote
        )
        if self._loaded_tokenizer.pad_token is None:
            self._loaded_tokenizer.pad_token = self._loaded_tokenizer.eos_token

    # --------------------------------------------------------------- generate

    def _build_steering_payload(self) -> List[Any]:
        from vllm_lens import SteeringVector

        payload: List[SteeringVector] = []
        for spec in self.steering_vectors:
            payload.append(
                SteeringVector(
                    activations=spec.vector.detach().to(torch.float32).cpu(),
                    layer_indices=list(spec.layer_indices),
                    scale=float(spec.scale),
                    norm_match=spec.norm_match,
                    position_indices=list(spec.position_indices) if spec.position_indices is not None else None,
                )
            )
        return payload

    def generate(self, prompts: List[str]) -> List[str]:
        self._ensure_loaded()
        from vllm import SamplingParams

        if self.chat_template:
            try:
                formatted = [
                    self._apply_chat_template(self._loaded_tokenizer, p) for p in prompts
                ]
            except Exception as e:
                logger.warning(
                    "VllmSteeredBackend: chat_template failed (%s); using raw prompts.", e
                )
                formatted = list(prompts)
        else:
            formatted = list(prompts)

        cfg = self.config
        sampling = SamplingParams(
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p if cfg.temperature > 0 else 1.0,
            repetition_penalty=cfg.repetition_penalty,
            stop=cfg.stop_sequences or None,
            seed=cfg.seed,
            extra_args={"apply_steering_vectors": self._build_steering_payload()},
        )
        outputs = self._loaded_llm.generate(formatted, sampling)
        return [self._truncate_on_stop(o.outputs[0].text) for o in outputs]


__all__ = ["VllmSteeredBackend", "SteeringSpec"]
