"""Harden runner — regularization-based trainers (SAP, AsFT, Surgery, Booster)."""
import os
import torch
from transformers import AutoModelForCausalLM

from ._base import (
    _HardenBase, _keep_model_columns, default_data_collator,
    _sap_safety_collator, _sap_contrastive_collator, _BFLOAT16, _to_dev,
)
from safetune.runner.utils.model_utils import free
import safetune.harden as HARD
from torch.utils.data import DataLoader

# ── SAPTrainer ────────────────────────────────────────────────────────────────

class SAPTrainer(_HardenBase):
    """SAP: contrastive alignment + perturbation.

    Args:
        grad_rate: gradient mixing rate. Default 0.1.
        v_update_step: virtual gradient update step size. Default 0.05.
        contrastive_temperature: contrastive loss temperature. Default 1.0.
    """

    METHOD = "SAP"

    def __init__(self, model=None, tokenizer=None, *,
                 grad_rate: float = 0.1,
                 v_update_step: float = 0.05,
                 contrastive_temperature: float = 1.0,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.grad_rate = grad_rate
        self.v_update_step = v_update_step
        self.contrastive_temperature = contrastive_temperature

    def train(self, train_dataset, out_dir: str = None, *,
              safety_dataset=None, **kwargs) -> str:
        # SAP's inner step maximizes a safe-useful *gap* from a contrastive
        # safety batch (safe `chosen_labels` vs harmful `rejected_labels`).
        # Default to a real contrastive set; the old build_safety_dataset +
        # _sap_safety_collator path masked rejected_labels to -100, so there was
        # no contrastive signal at all.
        contrastive = True
        if safety_dataset is None:
            from safetune.runner.utils.data_utils import sap_contrastive_dataset
            safety_dataset = sap_contrastive_dataset(self.tok)
        else:
            cols = getattr(safety_dataset, "column_names", None) or []
            contrastive = "chosen_labels" in cols and "rejected_labels" in cols
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        args = self._configure_args(
            HARD.SAPConfig(grad_rate=self.grad_rate, v_update_step=self.v_update_step),
            out_dir)

        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        if hasattr(safety_dataset, "with_format"):
            safety_dataset = safety_dataset.with_format("torch")

        if contrastive:
            # Keep the contrastive label columns (_keep_model_columns would strip
            # chosen_labels/rejected_labels).
            safety_loader = DataLoader(
                safety_dataset, batch_size=self.batch_size, shuffle=True,
                collate_fn=_sap_contrastive_collator)
        else:
            safety_loader = DataLoader(
                _keep_model_columns(safety_dataset), batch_size=self.batch_size,
                shuffle=True, collate_fn=_sap_safety_collator)
        tr = HARD.SAPTrainer(
            model=model, args=args,
            train_dataset=_keep_model_columns(train_dataset),
            data_collator=default_data_collator,
            safety_dataloader=safety_loader,
            contrastive_temperature=self.contrastive_temperature)
        tr.train()
        return self._save_merged(model, out_dir)

# ── AsFTTrainer ───────────────────────────────────────────────────────────────

class AsFTTrainer(_HardenBase):
    """AsFT: safety-basin orthogonal penalty constraint.

    Args:
        reg_lambda: regularization coefficient. Default 1.0.
        aligned_model_path: HF path/ID for the aligned reference.
    """

    METHOD = "AsFT"

    def __init__(self, model=None, tokenizer=None, *,
                 reg_lambda: float = 1.0,
                 aligned_model_path: str = None,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.reg_lambda = reg_lambda
        self.aligned_model_path = aligned_model_path

    def train(self, train_dataset, out_dir: str = None, **kwargs) -> str:
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        base_obj = model.get_base_model() if hasattr(model, "get_base_model") else model
        base_sd = {k: v.detach().cpu().clone()
                   for k, v in base_obj.state_dict().items()}
        if not self.aligned_model_path:
            import logging
            logging.getLogger(__name__).warning(
                "AsFTTrainer: no aligned_model_path given — the aligned "
                "reference defaults to the training model, so the alignment "
                "direction is zero and AsFT reduces to plain LoRA SFT. Pass "
                "aligned_model_path=<pre-drift aligned checkpoint>."
            )
        aligned_path = (self.aligned_model_path
                        or getattr(self.tok, "name_or_path", None))
        aligned_obj = AutoModelForCausalLM.from_pretrained(
            aligned_path, torch_dtype=_BFLOAT16)
        aligned_sd = {k: v.detach().cpu().clone()
                      for k, v in aligned_obj.state_dict().items()}
        del aligned_obj; free()
        args = self._configure_args(HARD.AsFTConfig(reg_lambda=self.reg_lambda), out_dir)
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        tr = HARD.AsFTTrainer(
            model=model, args=args,
            train_dataset=_keep_model_columns(train_dataset),
            data_collator=default_data_collator,
            aligned_state_dict=aligned_sd,
            base_state_dict=base_sd)
        tr.train()
        return self._save_merged(model, out_dir)

# ── SurgeryTrainer ────────────────────────────────────────────────────────────

class SurgeryTrainer(_HardenBase):
    """Surgery: surgical gradient modification (refusal-aware).

    Args:
        sink_lambda: refusal anchor loss weight. Default 0.01.
    """

    METHOD = "Surgery"

    def __init__(self, model=None, tokenizer=None, *,
                 sink_lambda: float = 0.01,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.sink_lambda = sink_lambda

    def train(self, train_dataset, out_dir: str = None, *,
              safety_dataset=None, **kwargs) -> str:
        if safety_dataset is None:
            from safetune.runner.utils.data_utils import build_safety_dataset
            safety_dataset = build_safety_dataset(self.tok)
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        args = self._configure_args(HARD.SurgeryConfig(sink_lambda=self.sink_lambda), out_dir)
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        if hasattr(safety_dataset, "with_format"):
            safety_dataset = safety_dataset.with_format("torch")
        refusal_loader = DataLoader(
            _keep_model_columns(safety_dataset), batch_size=self.batch_size, shuffle=True,
            collate_fn=default_data_collator)
        tr = HARD.SurgeryTrainer(
            model=model, args=args,
            train_dataset=_keep_model_columns(train_dataset),
            data_collator=default_data_collator,
            refusal_dataset=refusal_loader)
        tr.train()
        return self._save_merged(model, out_dir)

# ── BoosterTrainer ────────────────────────────────────────────────────────────

class BoosterTrainer(_HardenBase):
    """Booster: finite-difference harm regularizer.

    Args:
        perturb_scale: perturbation scale for finite-difference. Default 0.01.
    """

    METHOD = "Booster"

    def __init__(self, model=None, tokenizer=None, *,
                 perturb_scale: float = 0.01,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.perturb_scale = perturb_scale

    def train(self, train_dataset, out_dir: str = None, *,
              harmful_batches=None, **kwargs) -> str:
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        cfg = HARD.BoosterConfig(alpha=self.perturb_scale)
        task_loss_fn = lambda m, b: m(**b).loss
        if harmful_batches is None:
            from safetune.runner.utils.data_utils import harden_contamination_sets
            harm_ds = harden_contamination_sets(self.tok, n=64)[0]
            dev = next(model.parameters()).device
            harmful_batches = [
                {k: torch.tensor([harm_ds[i][k]]).to(dev)
                 for k in ("input_ids", "attention_mask", "labels")}
                for i in range(min(len(harm_ds), 8))
            ]
        # Compute harmful gradient g_h once at initial weights.
        harm_grads = HARD.collect_harmful_gradient(model, harmful_batches, task_loss_fn)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        loader = DataLoader(_keep_model_columns(train_dataset), batch_size=self.batch_size,
                            shuffle=True, collate_fn=default_data_collator)
        model.train()
        for _ in range(self.epochs):
            for batch in loader:
                batch = _to_dev(batch, model.device)
                with HARD.booster_simulated_perturbation(model, harm_grads, cfg):
                    perturbed = HARD.collect_harmful_gradient(model, harmful_batches,
                                                              task_loss_fn)
                model.zero_grad(set_to_none=True)
                loss = task_loss_fn(model, batch)
                loss.backward()
                align_grads = {n: p.grad.detach().clone()
                               for n, p in model.named_parameters()
                               if p.grad is not None}
                model.zero_grad(set_to_none=True)
                proj = HARD.booster_project(align_grads, harm_grads, perturbed, cfg)
                for n_, p in model.named_parameters():
                    g = proj.get(n_)
                    if g is not None:
                        p.grad = g.to(p.dtype).to(p.device)
                opt.step()
                opt.zero_grad(set_to_none=True)
        return self._save_merged(model, out_dir)

