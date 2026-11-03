"""Harden runner — regularization-based trainers (SAP, AsFT, Surgery, Booster)."""
import os
import torch
from safetune._refusal_helpers import _load_pretrained_lm

from ._base import (
    _HardenBase, _keep_model_columns, default_data_collator,
    _sap_safety_collator, _sap_contrastive_collator, _BFLOAT16, _to_dev,
)
from safetune.runner.utils.model_utils import free
from safetune.config import resolve_dtype
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
    HF_TRAINER = HARD.SAPHFTrainer

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
        tr = HARD.SAPHFTrainer(
            model=model, args=args,
            train_dataset=_keep_model_columns(train_dataset),
            data_collator=default_data_collator,
            safety_dataloader=safety_loader,
            contrastive_temperature=self.contrastive_temperature)
        tr.train()
        return self._save_merged(model, out_dir)

# ── AsFTTrainer ───────────────────────────────────────────────────────────────

class AsFTTrainer(_HardenBase):
    """AsFT: safety-basin orthogonal penalty constraint (Yang et al., arXiv:2506.08473).

    The model being trained is the **aligned** (safety-tuned) model, as in the
    authors' script. The alignment direction of every LoRA-targeted matrix is
    ``d = W_aligned - W_base``, where ``W_aligned`` is the trained model's weight
    at construction and ``W_base`` comes from ``base_model_path``.

    Args:
        reg_lambda: regularization coefficient. Default 1.0.
        base_model_path: HF path/ID of the pre-alignment base model the aligned
            model was tuned from. Loaded on CPU, only the LoRA-targeted weights
            are kept, then freed. Without it the run is plain LoRA SFT (a
            warning is logged).
        aligned_model_path: removed; raises ``ValueError``. It used to be the
            "aligned reference" but the direction was ``aligned_path - trained
            model``, which is neither the authors' convention nor useful.
    """

    METHOD = "AsFT"
    HF_TRAINER = HARD.AsFTHFTrainer

    def __init__(self, model=None, tokenizer=None, *,
                 reg_lambda: float = 1.0,
                 base_model_path: str = None,
                 aligned_model_path: str = None,
                 **kwargs):
        if aligned_model_path is not None:
            raise ValueError(
                "AsFTTrainer: aligned_model_path is no longer accepted. The model being "
                "trained is the aligned model; pass the pre-alignment base model as "
                "base_model_path (the direction is W_aligned - W_base, as in the AsFT "
                "authors' script)."
            )
        super().__init__(model, tokenizer, **kwargs)
        self.reg_lambda = reg_lambda
        self.base_model_path = base_model_path

    @staticmethod
    def _lora_target_weights(model):
        """``{plain module name + ".weight": CPU copy}`` of every LoRA-wrapped linear."""
        out = {}
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "base_layer"):
                key = name[len("base_model.model."):] if name.startswith("base_model.model.") else name
                out[key + ".weight"] = mod.base_layer.weight.detach().cpu().clone()
        return out

    def train(self, train_dataset, out_dir: str = None, **kwargs) -> str:
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        # W_aligned: the trained model's weights, copied before any update.
        aligned_sd = base_sd = None
        if self.base_model_path:
            aligned_sd = self._lora_target_weights(model)
            # Same dtype as the trained weights, so W_aligned - W_base is not rounding noise.
            dtype = next(iter(aligned_sd.values())).dtype if aligned_sd else resolve_dtype()
            base_obj = _load_pretrained_lm(
                self.base_model_path, torch_dtype=dtype, device_map="cpu")
            base_full = base_obj.state_dict()
            base_sd = {k: base_full[k].detach().cpu().clone()
                       for k in aligned_sd if k in base_full}
            del base_obj, base_full; free()
        else:
            import logging
            logging.getLogger(__name__).warning(
                "AsFTTrainer: no base_model_path given — there is no alignment "
                "direction, so AsFT reduces to plain LoRA SFT. Pass "
                "base_model_path=<pre-alignment base model>."
            )
        args = self._configure_args(HARD.AsFTConfig(reg_lambda=self.reg_lambda), out_dir)
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        tr = HARD.AsFTHFTrainer(
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
        sink_lambda: weight lambda of the attention-sink divergence regularizer (Eq. 4 of
            arXiv:2602.05228). Default 300, as in the paper and the official script.
    """

    METHOD = "Surgery"
    HF_TRAINER = HARD.SurgeryHFTrainer

    def __init__(self, model=None, tokenizer=None, *,
                 sink_lambda: float = 300.0,
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
        # The sink penalty reads the attention weights, which sdpa / flash attention do not
        # return. SurgeryHFTrainer still raises if this has no effect on the model.
        model.set_attn_implementation("eager")
        args = self._configure_args(HARD.SurgeryConfig(sink_lambda=self.sink_lambda), out_dir)
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        if hasattr(safety_dataset, "with_format"):
            safety_dataset = safety_dataset.with_format("torch")
        refusal_loader = DataLoader(
            _keep_model_columns(safety_dataset), batch_size=self.batch_size, shuffle=True,
            collate_fn=default_data_collator)
        tr = HARD.SurgeryHFTrainer(
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
        n_harmful_batches: harmful examples used for the harm gradient when
            ``harmful_batches`` is not passed. Default 8.
    """

    METHOD = "Booster"

    def __init__(self, model=None, tokenizer=None, *,
                 perturb_scale: float = 0.01,
                 n_harmful_batches: int = 8,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.perturb_scale = perturb_scale
        self.n_harmful_batches = n_harmful_batches

    def train(self, train_dataset, out_dir: str = None, *,
              harmful_batches=None, **kwargs) -> str:
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        cfg = HARD.BoosterConfig(alpha=self.perturb_scale)
        task_loss_fn = lambda m, b: m(**b).loss
        if harmful_batches is None:
            from safetune.runner.utils.data_utils import harden_contamination_sets
            # The first rows are the same for any n >= n_harmful_batches, so this
            # shares the other harden fallbacks' cached set (previously n=64).
            harm_ds = harden_contamination_sets(self.tok)[0]
            dev = next(model.parameters()).device
            harmful_batches = [
                {k: torch.tensor([harm_ds[i][k]]).to(dev)
                 for k in ("input_ids", "attention_mask", "labels")}
                for i in range(min(len(harm_ds), self.n_harmful_batches))
            ]
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        loader = DataLoader(_keep_model_columns(train_dataset), batch_size=self.batch_size,
                            shuffle=True, collate_fn=default_data_collator)
        model.train()
        for _ in range(self.epochs):
            for batch in loader:
                batch = _to_dev(batch, model.device)
                # g_h at the CURRENT weights (git-disl/Booster trainer.py:281-291), then g_h' at
                # w - alpha * g_h / ||g_h|| on the same harmful batches (trainer.py:293-310).
                harm_grads = HARD.collect_harmful_gradient(model, harmful_batches, task_loss_fn)
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

