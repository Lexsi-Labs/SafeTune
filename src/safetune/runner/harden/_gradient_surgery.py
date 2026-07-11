"""Harden runner — gradient-surgery trainers (PlainSFT baseline + SafeGrad)."""
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, Trainer

from ._base import (
    _HardenBase, _keep_model_columns, default_data_collator, _BFLOAT16,
)
import safetune.harden as HARD
from safetune.runner.utils.model_utils import free


# ── PlainSFTTrainer ───────────────────────────────────────────────────────────

class PlainSFTTrainer(_HardenBase):
    """Undefended SFT baseline on contaminated data (no defense)."""

    METHOD = "PlainSFT"

    def train(self, train_dataset, out_dir: str = None, **kwargs) -> str:
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()

        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")

        tr = Trainer(model=model,
                     args=self._training_args(out_dir),
                     train_dataset=_keep_model_columns(train_dataset),
                     data_collator=default_data_collator)
        tr.train()
        return self._save_merged(model, out_dir)

# ── SafeGradTrainer ───────────────────────────────────────────────────────────

class SafeGradTrainer(_HardenBase):
    """SafeGrad: gradient surgery + KL alignment vs frozen reference.

    Args:
        rho: gradient surgery mixing weight. Default 1.0.
        kl_temperature: KL alignment temperature. Default 1.0.
        reference_model_path: HF path/ID for the reference model.
    """

    METHOD = "SafeGrad"

    def __init__(self, model=None, tokenizer=None, *,
                 rho: float = 1.0,
                 kl_temperature: float = 1.0,
                 reference_model_path: str = None,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.rho = rho
        self.kl_temperature = kl_temperature
        self.reference_model_path = reference_model_path

    def train(self, train_dataset, out_dir: str = None, *,
              safety_dataset=None, **kwargs) -> str:
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        dev = next(model.parameters()).device
        ref_path = self.reference_model_path or getattr(self.tok, "name_or_path", None)
        # Reference on the model's device/dtype — device-agnostic (CUDA / MPS / CPU).
        reference = AutoModelForCausalLM.from_pretrained(
            ref_path, torch_dtype=next(model.parameters()).dtype).to(dev).eval()

        if safety_dataset is None:
            safety_dataset = self._build_safety_ds()

        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        if hasattr(safety_dataset, "with_format"):
            safety_dataset = safety_dataset.with_format("torch")
            
        safety_loader = DataLoader(
            _keep_model_columns(safety_dataset) if safety_dataset is not None
            else self._build_safety_ds(),
            batch_size=self.batch_size, shuffle=True,
            collate_fn=default_data_collator)
        tr = HARD.SafeGradTrainer(
            model=model, args=self._training_args(out_dir),
            train_dataset=_keep_model_columns(train_dataset),
            data_collator=default_data_collator,
            safety_dataset=safety_loader,
            reference_model=reference,
            rho=self.rho,
            kl_temperature=self.kl_temperature)
        tr.train()
        del reference; free()
        return self._save_merged(model, out_dir)

    def _build_safety_ds(self):
        from safetune.runner.utils.data_utils import build_safety_dataset
        return build_safety_dataset(self.tok)

