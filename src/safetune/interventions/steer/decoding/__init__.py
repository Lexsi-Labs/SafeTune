"""
Decoding-time safety steering: logit-level interventions during generation.

Distinct from the hidden-state / activation-level steering in
``safetune.steer``: these methods modify the *logits* a model produces at
each generation step, optionally using a second "safety guide" model as a
reference. They compose with HF's ``transformers.LogitsProcessor`` so they
slot into any ``model.generate()`` call.

Methods:

* :class:`ContrastiveDecodingProcessor` (Li et al., 2023): boosts tokens
  where strong model > weak model (or here, target model > weak base model).
* :class:`ProxyTuningProcessor` (Liu et al., 2024): apply the logit delta
  between a small fine-tuned model and its base to a third (larger) model,
  shifting it toward the small model's behavior without retraining.
* :class:`SafeDecodingProcessor` (Xu et al., ACL 2024): blend target logits
  with a safety-tuned guide at the shared vocabulary intersection. Per-step
  alpha schedule that decays linearly across the first m tokens.
* :class:`NudgingProcessor` (Fei et al., 2025): on high-entropy steps, swap
  to a small aligned guide; on low-entropy steps, defer to the target. The
  guide's vocabulary may differ; the processor aligns by token id where
  possible.

All four take a ``guide_model`` keyword that runs alongside the target. Pass
``backend="vllm"`` to use vLLM for both; for vLLM-Lens activation-level
steering, see :class:`safetune.core.eval.pipeline.backends.VllmSteeredBackend`.
"""
from .contrastive import ContrastiveDecodingConfig, ContrastiveDecodingProcessor
from .proxy_tuning import ProxyTuningConfig, ProxyTuningProcessor
from .safedecoding import SafeDecodingConfig, SafeDecodingProcessor
from .nudging import NudgingConfig, NudgingProcessor

__all__ = [
    "ContrastiveDecodingConfig",
    "ContrastiveDecodingProcessor",
    "ProxyTuningConfig",
    "ProxyTuningProcessor",
    "SafeDecodingConfig",
    "SafeDecodingProcessor",
    "NudgingConfig",
    "NudgingProcessor",
]
