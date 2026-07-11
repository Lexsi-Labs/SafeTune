"""Harden runner — high-level API for training-time safety hardening.

Usage::

    from safetune.runner import harden

    trainer = harden.LisaTrainer(model, tokenizer, lisa_rho=0.1)
    out_path = trainer.train(train_dataset, out_dir="/tmp/lisa_ckpt",
                             safety_dataset=safety_ds)
    metrics = trainer.eval("lisa_ckpt", out_path)
    trainer.save_results(metrics, variant="default")
"""
from __future__ import annotations
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from ._gradient_surgery import PlainSFTTrainer, SafeGradTrainer
from ._data_shaping import (
    LisaTrainer,
    SPPFTTrainer,
    LookAheadTrainer,
    STARDSSTrainer,
    DeRTaTrainer,
)
from safetune.harden import CSTTrainer
from ._regularization import (
    AsFTTrainer,
    SAPTrainer,
    SurgeryTrainer,
    BoosterTrainer,
)
from ._representation import (
    VaccineTrainer,
    TVaccineTrainer,
)
from ._tamper_resistant import (
    RepNoiseTrainer,
    CTRAPTrainer,
    SEAMTrainer,
    DOORTrainer,
)
from safetune.harden import MARTTrainer, DeepRefusalTrainer, AntibodyTrainer
from ._other import (
    TARTrainer,
    SaLoRATrainer,
    SEALTrainer,
    ConstrainedSFTTrainer,
    LoXHardenTrainer,
)


def load_harden_data(model_id, n: int = 512, max_len: int = 256):
    from safetune.runner.utils.dataset import load_harden_dataset
    from safetune.runner.utils.model_utils import load_tok
    return load_harden_dataset(load_tok(model_id), n=n, max_len=max_len)


__all__ = [
    "PlainSFTTrainer",
    "SafeGradTrainer",
    "LisaTrainer",
    "SPPFTTrainer",
    "LookAheadTrainer",
    "STARDSSTrainer",
    "DeRTaTrainer",
    "AsFTTrainer",
    "SAPTrainer",
    "SurgeryTrainer",
    "BoosterTrainer",
    "VaccineTrainer",
    "TVaccineTrainer",
    "RepNoiseTrainer",
    "CTRAPTrainer",
    "SEAMTrainer",
    "DOORTrainer",
    "TARTrainer",
    "SaLoRATrainer",
    "SEALTrainer",
    "ConstrainedSFTTrainer",
    "LoXHardenTrainer",
    "CSTTrainer",
    "MARTTrainer",
    "DeepRefusalTrainer",
    "AntibodyTrainer",
    "load_harden_data",
]
