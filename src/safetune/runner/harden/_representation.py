"""Harden runner — representation-perturbation trainers (Vaccine, TVaccine)."""
import os
import torch
from transformers import AutoModelForCausalLM

from ._base import _HardenBase, _keep_model_columns, default_data_collator, _to_dev
import safetune.harden as HARD
from torch.utils.data import DataLoader

# ── VaccineTrainer ────────────────────────────────────────────────────────────

class VaccineTrainer(_HardenBase):
    """Vaccine: SAM-style perturbation vaccine against backdoor attacks.

    Args:
        rho: SAM perturbation radius. Default 2.0.
    """

    METHOD = "Vaccine"

    def __init__(self, model=None, tokenizer=None, *, rho: float = 2.0, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.rho = rho

    def train(self, train_dataset, out_dir: str = None, **kwargs) -> str:
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        cfg = HARD.VaccineConfig(rho=self.rho)
        task_loss_fn = lambda m, b: m(**b).loss
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        loader = DataLoader(_keep_model_columns(train_dataset), batch_size=self.batch_size,
                            shuffle=True, collate_fn=default_data_collator)
        model.train()
        for _ in range(self.epochs):
            for batch in loader:
                batch = _to_dev(batch, model.device)
                loss = HARD.vaccine_loss(model, batch, task_loss_fn, cfg)
                loss.backward()
                opt.step()
                opt.zero_grad()
        return self._save_merged(model, out_dir)

# ── TVaccineTrainer ───────────────────────────────────────────────────────────

class TVaccineTrainer(_HardenBase):
    """T-Vaccine: layer-selective SAM perturbation.

    Args:
        rho: SAM perturbation radius. Default 2.0.
        top_k_ratio: fraction of top-k layers to perturb. Default 0.5.
    """

    METHOD = "TVaccine"

    def __init__(self, model=None, tokenizer=None, *,
                 rho: float = 2.0,
                 top_k_ratio: float = 0.5,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.rho = rho
        self.top_k_ratio = top_k_ratio

    def train(self, train_dataset, out_dir: str = None, **kwargs) -> str:
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        cfg = HARD.TVaccineConfig(rho=self.rho, top_k_ratio=self.top_k_ratio)
        task_loss_fn = lambda m, b: m(**b).loss
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        loader = DataLoader(_keep_model_columns(train_dataset), batch_size=self.batch_size,
                            shuffle=True, collate_fn=default_data_collator)
        model.train()
        for _ in range(self.epochs):
            for batch in loader:
                batch = _to_dev(batch, model.device)
                loss = HARD.tvaccine_loss(model, batch, task_loss_fn, cfg)
                loss.backward()
                opt.step()
                opt.zero_grad()
        return self._save_merged(model, out_dir)

