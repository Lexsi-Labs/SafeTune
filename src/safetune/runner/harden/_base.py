"""Harden runner — base class and shared helpers."""
from __future__ import annotations
import dataclasses
import functools
import gc
import logging
import os
import warnings
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, default_data_collator

from safetune.runner.utils.eval_runner import eval_safety, eval_utility, all_metrics
from safetune.runner.utils.results_writer import ResultsWriter, DEFAULT_RESULTS_DIR
from safetune.runner.utils.model_utils import lora_wrap, free, derive_model_id
from safetune.config import get_config, resolve_device, resolve_dtype, warn_unknown_kwargs

logger = logging.getLogger(__name__)

_BFLOAT16 = torch.bfloat16
_LORA_METHODS = {"NPO", "FLAT", "SimDPO"}


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _keep_model_columns(ds):
    """Drop all dataset columns except input_ids / attention_mask / labels."""
    return ds.remove_columns([
        c for c in ds.column_names
        if c not in ("input_ids", "attention_mask", "labels")
    ])


def _accepts_raw_data(train):
    """Wrap a trainer's ``train()`` so ``train_dataset`` and ``safety_dataset`` may be
    a raw ``Dataset`` (tokenized here with the trainer's tokenizer) or a source
    string: a table name, HF id, local file or dataset folder, with
    ``dataset_config=`` / ``safety_dataset_config=`` (e.g. a CuratorKIT export and
    ``"sft_sharegpt"``) and ``dataset_split=`` / ``safety_dataset_split=``
    (default ``train``). Source strings are recorded in ``lexsi_provenance.json``."""
    @functools.wraps(train)
    def wrapper(self, train_dataset, *args, dataset_config=None, dataset_split=None,
                safety_dataset_config=None, safety_dataset_split=None, **kwargs):
        train_dataset = self._prepare_dataset(train_dataset, dataset_config, dataset_split)
        if kwargs.get("safety_dataset") is not None:
            kwargs["safety_dataset"] = self._prepare_dataset(
                kwargs["safety_dataset"], safety_dataset_config, safety_dataset_split)
        return train(self, train_dataset, *args, **kwargs)
    return wrapper


# ── Training-args helpers ─────────────────────────────────────────────────────

def _precision(bf16=None, fp16=None) -> dict:
    """``bf16`` / ``fp16`` / ``use_cpu`` (and ``seed`` if configured) for
    TrainingArguments on the runtime device.

    ``None`` for both derives precision from the runtime dtype (bf16 where
    supported). A requested precision the device can't run (bf16 on CPU or a
    pre-Ampere GPU, fp16 off CUDA) falls back with a warning instead of letting
    TrainingArguments raise.
    """
    device = resolve_device()
    supported = resolve_dtype("auto", device)
    if bf16 is None and fp16 is None:
        want = resolve_dtype(device=device)
    else:
        want = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
    ok = {torch.float32: True, torch.bfloat16: supported == torch.bfloat16,
          torch.float16: device.startswith("cuda")}
    if not ok.get(want, False):
        logger.warning("%s training is not supported on %s; using %s instead.",
                       want, device, supported)
        want = supported
    out = dict(bf16=want == torch.bfloat16, fp16=want == torch.float16,
               use_cpu=device == "cpu")
    seed = get_config().seed
    if seed is not None:
        out["seed"] = seed
    return out


def _standard_args(out_dir, epochs=1, batch_size=4, lr=1e-4, bf16=None,
                   optimizer="adamw_torch", logging_steps=10, fp16=None, wandb=False) -> dict:
    """The TrainingArguments fields every SafeTune harden trainer sets."""
    return dict(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=logging_steps,
        save_strategy="no",
        report_to=(["wandb"] if wandb else []),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim=optimizer,
        **_precision(bf16, fp16),
    )


# Fields _standard_args always sets: user kwargs with these names are not
# forwarded to method configs (the trainer's own epochs / batch_size / lr /
# optimizer / out_dir and the runtime config win).
_STAMPED = frozenset({
    "output_dir", "num_train_epochs", "per_device_train_batch_size", "learning_rate",
    "logging_steps", "save_strategy", "report_to", "dataloader_num_workers",
    "remove_unused_columns", "optim", "bf16", "fp16", "use_cpu", "seed",
})


def _make_training_args(out_dir, epochs=1, batch_size=4, lr=1e-4,
                        bf16=None, optimizer="adamw_torch", logging_steps=10,
                        fp16=None, wandb=False):
    """Build a bare TrainingArguments with the standard SafeTune defaults."""
    from transformers import TrainingArguments
    return TrainingArguments(**_standard_args(
        out_dir, epochs, batch_size, lr, bf16, optimizer, logging_steps, fp16, wandb))


def _apply_training_args(config, out_dir, epochs=1, batch_size=4, lr=1e-4,
                         bf16=None, optimizer="adamw_torch", logging_steps=10,
                         fp16=None, wandb=False):
    """A copy of ``config`` with the standard SafeTune training settings.

    Rebuilt with ``dataclasses.replace`` rather than setattr so TrainingArguments
    re-validates and picks its device from the new ``use_cpu`` (the device is
    cached at construction).
    """
    return dataclasses.replace(config, **_standard_args(
        out_dir, epochs, batch_size, lr, bf16, optimizer, logging_steps, fp16, wandb))


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


# Arguments only the transformers.Trainer form takes; no runner trainer accepts them.
_HF_ONLY_KWARGS = frozenset({"args", "train_dataset", "eval_dataset", "data_collator",
                             "processing_class", "ref_model"})


def _to_dev(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


# ── Base class ────────────────────────────────────────────────────────────────

class _HardenBase:
    PILLAR = "harden"
    METHOD: str = ""
    # Method config class (a TrainingArguments subclass) that user kwargs matching
    # its fields are forwarded to, via _method_config().
    CONFIG = None
    # The transformers.Trainer subclass this trainer runs, if any.
    HF_TRAINER = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "train" in cls.__dict__:  # every trainer's own train() takes raw data too
            cls.train = _accepts_raw_data(cls.__dict__["train"])

    def __new__(cls, *args, **kwargs):
        # Until 0.1.3, safetune.harden.<Name>Trainer was the HF Trainer subclass,
        # called as <Name>Trainer(model=, args=TrainingArguments, train_dataset=...).
        # Route that form to the subclass, now <Name>HFTrainer, for one release.
        if cls.HF_TRAINER is not None and (
                _HF_ONLY_KWARGS & kwargs.keys()
                or any(isinstance(a, TrainingArguments) for a in args[1:])):
            warnings.warn(
                f"{cls.__name__}(model=, args=, train_dataset=, ...) is the old "
                f"transformers.Trainer form; use safetune.harden.{cls.HF_TRAINER.__name__} "
                f"for that. The old form stops working in 0.3. {cls.__name__} is now the "
                f"high-level trainer: {cls.__name__}(model, tokenizer).train(dataset).",
                DeprecationWarning, stacklevel=2)
            return cls.HF_TRAINER(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        model=None,
        tokenizer=None,
        *,
        model_id: str = None,
        epochs: int = 1,
        batch_size: int = 4,
        lr: float = 1e-4,
        bf16: Optional[bool] = None,
        fp16: Optional[bool] = None,
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
        # inputs[] entries for the checkpoint's lexsi_provenance.json
        self.dataset_inputs: list = []
        warn_unknown_kwargs(self, kwargs, stamped=_STAMPED)

    def _prepare_dataset(self, dataset, config=None, split=None):
        """A source string loaded via ``safetune.data.dataset_ids.load``, then raw
        rows tokenized (``data_utils.tokenize_dataset``)."""
        from safetune.runner.utils.data_utils import tokenize_dataset
        if isinstance(dataset, (str, os.PathLike)):
            from safetune.data.dataset_ids import load
            from safetune.provenance import input_entry
            self.dataset_inputs.append(input_entry("dataset", dataset, config))
            dataset = load(str(dataset), config=config, split=split)
        elif config is not None or split is not None:
            raise TypeError("dataset_config / dataset_split need a dataset source string, "
                            "not a loaded dataset.")
        if "input_ids" not in (getattr(dataset, "column_names", None) or ["input_ids"]):
            dataset = tokenize_dataset(dataset, self.tok, max_len=get_config().max_len)
        return dataset

    def _method_config(self, **fixed):
        """``self.CONFIG`` built from ``fixed`` plus user kwargs matching its fields
        (except the standard fields the trainer sets itself, see ``_STAMPED``).

        The runtime precision (``bf16`` / ``fp16`` / ``use_cpu``) goes in at
        construction: TrainingArguments validates it in ``__post_init__``, so a
        config whose own default the device can't run (TRL's bf16) would raise
        before ``_configure_args`` could stamp it."""
        names = {f.name for f in dataclasses.fields(self.CONFIG)} - _STAMPED
        fixed.update({k: v for k, v in self._extra.items() if k in names})
        return self.CONFIG(**{**_precision(self.bf16, self.fp16), **fixed})

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
        if get_config().seed is not None:
            from transformers import set_seed
            set_seed(get_config().seed)  # custom training loops shuffle with torch's RNG
        return lora_wrap(m)

    def _save_merged(self, model, out_dir: str) -> str:
        """Merge LoRA adapter (if present) and save checkpoint; return the path."""
        from safetune.runner.utils.model_utils import save_checkpoint
        merged = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
        return save_checkpoint(merged, self.tok, os.path.basename(out_dir),
                               out_dir=os.path.dirname(out_dir),
                               method=f"{self.PILLAR}.{self.METHOD}",
                               inputs=self.dataset_inputs,
                               params=dict(epochs=self.epochs, batch_size=self.batch_size,
                                           lr=self.lr))

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
        gpu_mem = (kwargs.pop("gpu_memory_utilization", None)
                   or get_config().gpu_memory_utilization or 0.75)
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
