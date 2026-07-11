"""Recover runner — high-level API for weight-space post-hoc safety patching.

Usage::

    from safetune.runner import recover

    trainer = recover.CThetaTrainer(model, base_model=base, aligned_model=aligned)
    patched = trainer.apply()
    ckpt_path = trainer.save_checkpoint(patched, tokenizer, "ctheta_ckpt")
    metrics = trainer.eval("ctheta_ckpt", ckpt_path, drift_task="gsm8k")
    trainer.save_results(metrics, variant="strength=1.0")
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from ._circuit import CThetaTrainer
from ._whole_model import (
    TaskArithmeticTrainer,
    PrePostMergeTrainer,
    SOMFTrainer,
    WiseFTTrainer,
)
from ._layer import (
    ReStaTrainer,
    SafeMergeTrainer,
    SafeDeltaTrainer,
    SafeLoRATrainer,
    QReSafeTrainer,
    AAQTrainer,
    RepNoiseRecoverTrainer,
)
from ._low_rank import (
    LoXTrainer,
    LSSFTrainer,
    SafetyVectorRestoreTrainer,
)
from ._neuron import (
    NLSRTrainer,
    AntidoteTrainer,
    AntidoteV2Trainer,
    MSCPTrainer,
)
from ._saliency import (
    GradSelectiveRecoverTrainer,
    OneShotSafetyPatchTrainer,
)
from ._other import (
    PKETrainer,
    SafeReActTrainer,
    SCRUBTrainer,
)


def load_recover_data(model_id, max_len: int = 64):
    """Tokenized calibration inputs for recover methods."""
    import torch
    from safetune.runner.utils.dataset import load_recover_dataset
    from safetune.runner.utils.model_utils import load_tok
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_recover_dataset(load_tok(model_id), device, max_len=max_len)


__all__ = [
    "CThetaTrainer",
    "TaskArithmeticTrainer",
    "PrePostMergeTrainer",
    "SOMFTrainer",
    "WiseFTTrainer",
    "ReStaTrainer",
    "SafeMergeTrainer",
    "SafeDeltaTrainer",
    "SafeLoRATrainer",
    "QReSafeTrainer",
    "AAQTrainer",
    "RepNoiseRecoverTrainer",
    "LoXTrainer",
    "LSSFTrainer",
    "SafetyVectorRestoreTrainer",
    "NLSRTrainer",
    "AntidoteTrainer",
    "AntidoteV2Trainer",
    "MSCPTrainer",
    "GradSelectiveRecoverTrainer",
    "OneShotSafetyPatchTrainer",
    "PKETrainer",
    "SafeReActTrainer",
    "SCRUBTrainer",
    "load_recover_data",
]
