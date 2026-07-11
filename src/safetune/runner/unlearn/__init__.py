"""Unlearn runner — high-level API for selective forget-set unlearning.

Usage::

    from safetune.runner import unlearn

    trainer = unlearn.RMUTrainer(
        model, layer_id=7, update_layer_ids=[5, 6, 7],
        max_num_batches=80, lr=5e-5,
    )
    unlearned = trainer.unlearn(forget=forget_ds, retain=retain_ds)
    ckpt_path  = trainer.save_checkpoint(unlearned, tokenizer, "rmu_ckpt")
    metrics    = trainer.eval("rmu_ckpt", ckpt_path, drift_task="gsm8k")
    trainer.save_results(metrics, variant="default")

Naming: ``MethodName``Trainer.

All trainers expose::

    .unlearn(forget, retain, *, **kwargs)  →  patched_model
    .eval(folder_name, model_path, *, drift_task=None, **kwargs)  →  dict
    .save_checkpoint(model, tokenizer, name)  →  str (path)
    .save_results(metrics, *, variant="default")  →  None
"""
from __future__ import annotations
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from ._rmu import RMUTrainer
from ._npo import NPOTrainer
from ._ga import GradientAscentTrainer, GradDiffTrainer
from ._flat import FLATTrainer
from ._simdpo import SimDPOTrainer


def load_unlearn_data(model_id, n: int = 256, max_len: int = 256):
    from safetune.runner.utils.dataset import load_unlearn_dataset
    from safetune.runner.utils.model_utils import load_tok
    return load_unlearn_dataset(load_tok(model_id), n=n, max_len=max_len)


__all__ = [
    "RMUTrainer",
    "NPOTrainer",
    "GradientAscentTrainer",
    "GradDiffTrainer",
    "FLATTrainer",
    "SimDPOTrainer",
    "load_unlearn_data",
]
