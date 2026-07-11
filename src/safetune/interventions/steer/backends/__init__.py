"""Inference backends for the STEER pillar.

A SafeTune steering method produces a *steered model* — a wrapper that installs
residual-stream hooks (activation steering) or a decoding-time logits processor.
How that steered model is *run* for generation is a separate concern, and that
is what a backend is:

* ``hf``          — plain ``transformers`` generation with the SafeTune hooks
                    installed (the reference path; always available).
* ``vllm-hook``   — activation steering served inside vLLM via the IBM/vLLM-Hook
                    worker (:class:`VLLMHookSteer`). ~6× faster than ``hf`` for
                    additive / projection steering methods.
* ``vllm-logits`` — decoding steering served inside vLLM via a V1 batch-level
                    ``LogitsProcessor`` (:class:`VLLMDecodeSteer`).

The unified entry point is :func:`safetune.steer.run`; see ``run.py``.
"""
from __future__ import annotations

# vLLM is an optional heavy dependency. Importing these adapter modules does not
# require vLLM (their vLLM imports are guarded), so they are safe to import here.
from .vllm_hook import (
    SteerSpec,
    extract_steer_spec,
    MultiLayerSteerWorker,
    register_safetune_worker,
    VLLMHookSteer,
)
from .vllm_logits import (
    DecodeSteerSpec,
    SafeTuneDecodeLogitsProcessor,
    VLLMDecodeSteer,
)
from .run import run

__all__ = [
    "run",
    "SteerSpec",
    "extract_steer_spec",
    "MultiLayerSteerWorker",
    "register_safetune_worker",
    "VLLMHookSteer",
    "DecodeSteerSpec",
    "SafeTuneDecodeLogitsProcessor",
    "VLLMDecodeSteer",
]
