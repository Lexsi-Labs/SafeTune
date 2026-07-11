"""Steer runner — high-level API for inference-time steering.

Usage::

    from safetune.runner import steer

    trainer = steer.RefusalDirectionTrainer(model, tokenizer)
    steered_model = trainer.calibrate(harmful=harmful, harmless=harmless)
    metrics = trainer.eval_live("refusal_dir_run", steered_model,
                                bench_prompts=bench_prompts)
    trainer.save_results(metrics, variant="default")

Naming: ``MethodName``Trainer  (matches safetune.steer class names).

Three sub-families:

  Static weight / materialized  — calibrate() extracts vectors/probes, then
                                  builds a wrapped model; can be serialised.
  Dynamic hook (HF native)      — steer_model() returns a wrapped model for
                                  live eval; vLLM-incompatible.
  Decoding / LogitsProcessor    — make_processor() returns a processor that
                                  is passed to HF generate().

All trainers expose::

    .calibrate(harmful, harmless, *, calib_n=256, **kwargs)
        -> wrapped_model  (or (wrapped_model, metadata))
    .eval_live(folder_name, wrapped_model, *, bench_prompts=None, **kwargs)
        -> dict  (safety + utility metrics)
    .save_results(metrics, *, variant="default")  ->  None
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from ._activation import (
    RefusalDirectionTrainer,
    CAATrainer,
    LinearProbeGuardTrainer,
    SCANSTrainer,
    STATrainer,
    CircuitBreakerRRTrainer,
    CASTTrainer,
    AdaSteerTrainer,
    SafeSwitchTrainer,
    CircuitBreakerTrainer,
    RepBendTrainer,
    TARSteerTrainer,
    RRFAEnsembleTrainer,
    SafeSteerTrainer,
    AlphaSteerTrainer,
)
from ._decoding import (
    SafeDecodingTrainer,
    ContrastiveDecodingTrainer,
    ProxyTuningTrainer,
    NudgingTrainer,
)


def load_steer_data(n: int = 256):
    from safetune.runner.utils.dataset import load_steer_dataset
    return load_steer_dataset(n=n)


__all__ = [
    "RefusalDirectionTrainer",
    "CAATrainer",
    "LinearProbeGuardTrainer",
    "SCANSTrainer",
    "STATrainer",
    "CircuitBreakerRRTrainer",
    "CASTTrainer",
    "AdaSteerTrainer",
    "SafeSwitchTrainer",
    "SafeDecodingTrainer",
    "ContrastiveDecodingTrainer",
    "ProxyTuningTrainer",
    "NudgingTrainer",
    "CircuitBreakerTrainer",
    "RepBendTrainer",
    "TARSteerTrainer",
    "RRFAEnsembleTrainer",
    "SafeSteerTrainer",
    "AlphaSteerTrainer",
    "load_steer_data",
]
