"""Declarative YAML configuration for SafeTune CLI runs.

Usage::

    # config.yaml
    command: train
    algo: lisa
    model: Qwen/Qwen2.5-0.5B-Instruct
    epochs: 3
    batch_size: 4
    lr: 2e-5
    train_dataset: beavertails
    train_split: 30k_train
    output: ./results/lisa
    # method-specific kwargs passed through to the trainer
    lisa_rho: 0.2
    lisa_warmup_steps: 20

    # CLI
    safetune --config config.yaml train --model other/model  # CLI flags override config

Any key not recognised as a standard field is forwarded to the trainer as a
keyword argument, making method-specific hyperparameters first-class citizens.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional

# Standard fields that must be numeric — coerced on load so a YAML string like
# "2e-5" (PyYAML parses `lr: 2e-5` as a str) never reaches the trainer.
_NUMERIC_FIELDS = {"epochs": int, "batch_size": int, "logging_steps": int, "lr": float}


@dataclass
class SafeTuneConfig:
    """Declarative config for a safetune run."""

    # Dispatch
    command: str = "train"
    algo: str = "safegrad"

    # Model
    model: str = ""
    base: Optional[str] = None
    aligned: Optional[str] = None

    # Output
    output: str = "./results"

    # Training
    epochs: int = 1
    batch_size: int = 1
    lr: float = 5e-5
    precision: str = "bf16"
    optimizer: str = "adamw_torch"
    logging_steps: int = 10

    # Dataset (harden / train)
    train_dataset: str = "beavertails"
    # None → dataset-aware default resolved in the CLI (30k_train for
    # beavertails, else 'train'); set explicitly to override.
    train_split: Optional[str] = None

    # Evaluation
    dataset: Optional[str] = None
    drift_task: Optional[str] = None
    # Generation backend for evaluation: "vllm" (fastest) or "hf" (transformers).
    # None → auto: use vLLM when it's installed, otherwise fall back to "hf" so
    # eval works on a fresh install with no vllm.
    eval_backend: Optional[str] = None

    # Method-specific kwargs (e.g. lisa_rho, rank, inner_steps, ...)
    method_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "SafeTuneConfig":
        """Load a SafeTuneConfig from a YAML file.

        Unknown keys are collected into ``method_kwargs`` and forwarded to the
        trainer, so method-specific hyperparameters need no special handling.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for --config support: pip install pyyaml"
            )
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}

        known = set(cls.__dataclass_fields__) - {"method_kwargs"}
        method_kwargs = {k: v for k, v in data.items() if k not in known}
        # PyYAML (1.1 spec) parses `lisa_rho: 1e-2` as the STRING "1e-2".
        # Coerce numeric-looking strings so trainers get numbers, same as the
        # standard-field coercion below.
        for k, v in method_kwargs.items():
            if isinstance(v, str):
                try:
                    method_kwargs[k] = int(v)
                except ValueError:
                    try:
                        method_kwargs[k] = float(v)
                    except ValueError:
                        pass  # genuinely a string
        # Coerce numeric fields: PyYAML parses e.g. `lr: 2e-5` as the *string*
        # "2e-5" (a 1.1-spec quirk), which would otherwise reach the trainer as a
        # string. A bad value raises loudly here rather than failing deep in training.
        standard = {}
        for k, v in data.items():
            if k not in known:
                continue
            if v is not None and k in _NUMERIC_FIELDS:
                v = _NUMERIC_FIELDS[k](v)
            standard[k] = v
        return cls(**standard, method_kwargs=method_kwargs)

    def to_namespace(self) -> argparse.Namespace:
        """Return an ``argparse.Namespace`` with all fields, including method_kwargs."""
        flat = {k: v for k, v in asdict(self).items() if k != "method_kwargs"}
        flat.update(self.method_kwargs)
        return argparse.Namespace(**flat)

    def as_trainer_kwargs(self) -> dict:
        """Return a dict suitable for passing as **kwargs to any Trainer constructor."""
        skip = {"command", "algo", "model", "base", "aligned", "output",
                "train_dataset", "train_split", "dataset", "method_kwargs"}
        kw = {k: v for k, v in asdict(self).items()
              if k not in skip and v is not None}
        kw.update(self.method_kwargs)
        return kw
