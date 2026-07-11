"""Harden runner — base class and shared helpers."""
from __future__ import annotations
import gc
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator

from safetune.runner.utils.eval_runner import eval_safety, eval_utility, all_metrics
from safetune.runner.utils.results_writer import ResultsWriter, DEFAULT_RESULTS_DIR
from safetune.runner.utils.model_utils import lora_wrap, free, derive_model_id

_BFLOAT16 = torch.bfloat16
_LORA_METHODS = {"NPO", "FLAT", "SimDPO"}


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _keep_model_columns(ds):
    """Drop all dataset columns except input_ids / attention_mask / labels."""
    return ds.remove_columns([
        c for c in ds.column_names
        if c not in ("input_ids", "attention_mask", "labels")
    ])


# ── Training-args helpers ─────────────────────────────────────────────────────

def _make_training_args(out_dir, epochs=1, batch_size=4, lr=1e-4,
                        bf16=True, optimizer="adamw_torch", logging_steps=10,
                        fp16=False, wandb=False):
    """Build a bare TrainingArguments with the standard SafeTune defaults."""
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=logging_steps,
        save_strategy="no",
        report_to=(["wandb"] if wandb else []),
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim=optimizer,
    )


def _apply_training_args(config, out_dir, epochs=1, batch_size=4, lr=1e-4,
                         bf16=True, optimizer="adamw_torch", logging_steps=10,
                         fp16=False, wandb=False):
    """Stamp the standard SafeTune training settings onto an existing config object."""
    for k, v in dict(
        output_dir=out_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, learning_rate=lr,
        logging_steps=logging_steps, save_strategy="no", bf16=bf16, fp16=fp16,
        report_to=(["wandb"] if wandb else []), remove_unused_columns=False,
        dataloader_num_workers=0, optim=optimizer,
    ).items():
        setattr(config, k, v)
    return config


# ── Collators ─────────────────────────────────────────────────────────────────

def _sap_safety_collator(features):
    batch = {}
    for k in ("input_ids", "attention_mask", "labels"):
        batch[k] = torch.stack([torch.as_tensor(f[k], dtype=torch.long) for f in features])
    batch["chosen_labels"] = batch["labels"].clone()
    batch["rejected_labels"] = torch.full_like(batch["labels"], -100)
    return batch


def _sap_contrastive_collator(features):
    """Collate a real contrastive SAP batch (safe vs harmful completion).

    Unlike ``_sap_safety_collator`` (which masks ``rejected_labels`` to -100 and
    so gives SAP no safe-useful gap to maximize), this keeps the distinct
    ``chosen_labels``/``rejected_labels`` produced by ``sap_contrastive_dataset``.
    """
    batch = {}
    for k in ("input_ids", "attention_mask", "chosen_labels", "rejected_labels"):
        batch[k] = torch.stack([torch.as_tensor(f[k], dtype=torch.long) for f in features])
    return batch


def _stardss_collator(features):
    batch = {}
    for k in ("input_ids", "attention_mask", "labels"):
        batch[k] = torch.stack([torch.as_tensor(f[k]) for f in features])
    batch["safety_weights"] = torch.stack(
        [torch.as_tensor(f["safety_weights"]) for f in features]
    )
    return batch


def _derta_collator(features):
    batch = {}
    for k in ("input_ids", "attention_mask", "labels"):
        batch[k] = torch.stack([torch.as_tensor(f[k], dtype=torch.long) for f in features])
    batch["safe"] = torch.tensor(
        [bool(f.get("safe", True)) for f in features], dtype=torch.bool
    )
    return batch


def _to_dev(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


# ── Base class ────────────────────────────────────────────────────────────────

class _HardenBase:
    PILLAR = "harden"
    METHOD: str = ""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        *,
        model_id: str = None,
        epochs: int = 1,
        batch_size: int = 4,
        lr: float = 1e-4,
        bf16: bool = True,
        fp16: bool = False,
        wandb: bool = False,
        optimizer: str = "adamw_torch",
        logging_steps: int = 10,
        results_dir: str = None,
        drift_task: str = None,
        **kwargs,
    ):
        self.model_id = derive_model_id(model_id, model, tokenizer)
        self.model = model
        self.tok = tokenizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.bf16 = bf16
        self.fp16 = fp16
        self.wandb = wandb
        self.optimizer = optimizer
        self.logging_steps = logging_steps
        self.results_dir = results_dir or DEFAULT_RESULTS_DIR
        self.drift_task = drift_task
        self._extra = kwargs

    @property
    def model(self):
        if getattr(self, '_model', None) is None:
            from safetune.runner.utils.model_utils import load_model
            self._model = load_model(self.model_id)
        return self._model

    @model.setter
    def model(self, v):
        self._model = v

    @property
    def tok(self):
        if getattr(self, '_tok', None) is None:
            from safetune.runner.utils.model_utils import load_tok
            self._tok = load_tok(self.model_id)
        return self._tok

    @tok.setter
    def tok(self, v):
        self._tok = v

    def _lora_base(self):
        if self._model is not None:
            m = self._model
            self._model = None
        else:
            from safetune.runner.utils.model_utils import load_model
            base_name = (getattr(self._tok, "name_or_path", None) if self._tok else None)
            if not base_name:
                mid = getattr(self, "model_id", None)
                base_name = mid if mid and mid != "model" else None
            if not base_name:
                raise ValueError(
                    "Cannot determine base model: pass model= or model_id= to the trainer."
                )
            m = load_model(base_name)
        return lora_wrap(m)

    def _save_merged(self, model, out_dir: str) -> str:
        """Merge LoRA adapter (if present) and save checkpoint; return the path."""
        from safetune.runner.utils.model_utils import save_checkpoint
        merged = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
        return save_checkpoint(merged, self.tok, os.path.basename(out_dir),
                               out_dir=os.path.dirname(out_dir))

    def _training_args(self, out_dir):
        """Build TrainingArguments using this trainer's configured settings."""
        return _make_training_args(
            out_dir, self.epochs, self.batch_size, self.lr,
            self.bf16, self.optimizer, self.logging_steps,
            fp16=self.fp16, wandb=self.wandb,
        )

    def _configure_args(self, config, out_dir):
        """Stamp this trainer's settings onto an existing config object."""
        return _apply_training_args(
            config, out_dir, self.epochs, self.batch_size, self.lr,
            self.bf16, self.optimizer, self.logging_steps,
            fp16=self.fp16, wandb=self.wandb,
        )

    def eval(
        self,
        folder_name: str,
        model_path: str,
        *,
        drift_task: str = None,
        **kwargs,
    ) -> dict:
        self._model = None
        self._tok = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        dt = drift_task or self.drift_task
        gpu_mem = kwargs.pop(
            "gpu_memory_utilization",
            float(os.environ.get("SAFETUNE_GPU_MEM", "0.75")),
        )
        eval_safety(folder_name, model_path, results_dir=self.results_dir,
                    gpu_memory_utilization=gpu_mem,
                    **{k: v for k, v in kwargs.items()
                       if k in ("gpu", "backend", "base_model_name")})
        eval_utility(folder_name, model_path, drift_task=dt,
                     results_dir=self.results_dir,
                     gpu_mem=gpu_mem,
                     **{k: v for k, v in kwargs.items()
                        if k in ("gpu", "backend", "base_model_name", "limit")})
        return all_metrics(folder_name, drift_task=dt, results_dir=self.results_dir)

    def save_results(
        self,
        metrics: dict,
        *,
        method: str = None,
        variant: str = "default",
    ) -> None:
        from safetune.runner.utils.eval_runner import safety_mean, utility_mean
        record = {
            "method": method or self.METHOD,
            "variant": variant,
            "metrics": metrics,
            "safety_mean": safety_mean(metrics),
            "utility_mean": utility_mean(metrics, drift_task=self.drift_task),
        }
        ResultsWriter(self.PILLAR, results_dir=self.results_dir).append(record)

    def evaluate(self, model_path: str, domain: str = None) -> dict:
        from safetune.runner.utils.eval_runner import safety_mean, utility_mean
        m = self.eval(os.path.basename(model_path), model_path, drift_task=domain)
        print(m)
        from safetune.runner.utils.eval_runner import fmt_metric
        print(f"[eval] {os.path.basename(model_path)}  safety={fmt_metric(safety_mean(m))}  utility={fmt_metric(utility_mean(m, drift_task=domain))}")
        return m

    def _resolve_out_dir(self, out_dir: str) -> str:
        if out_dir is None:
            out_dir = self.METHOD.lower() or "harden"
        if not os.path.isabs(out_dir) and os.sep not in out_dir and "/" not in out_dir:
            out_dir = os.path.join(self.results_dir, "checkpoints", out_dir)
        return out_dir
