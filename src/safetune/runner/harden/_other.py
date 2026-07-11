"""Harden runner — miscellaneous trainers (TAR, SaLoRA, SEAL, ConstrainedSFT, LoXHarden)."""
import os
import torch
import itertools
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, Trainer

from ._base import (
    _HardenBase, _keep_model_columns, default_data_collator, _BFLOAT16, _to_dev,
)
import safetune.harden as HARD
from safetune.runner.utils.model_utils import free


# ── TARTrainer ────────────────────────────────────────────────────────────────

class TARTrainer(_HardenBase):
    """TAR: task-aware regularization with inner-loop adversarial updates.

    Args:
        inner_steps: inner-loop optimization steps. Default 25.
        inner_lr: inner-loop learning rate. Default 1e-4.
        lambda_tar: TAR regularization coefficient. Default 1.0.
    """

    METHOD = "TAR"

    def __init__(self, model=None, tokenizer=None, *,
                 inner_steps: int = 25,
                 inner_lr: float = 1e-4,
                 lambda_tar: float = 1.0,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.lambda_tar = lambda_tar

    def train(self, train_dataset, out_dir: str = None, *,
              harm_dataset=None, safety_dataset=None, **kwargs) -> str:
        from tqdm import tqdm
        if harm_dataset is None:
            from safetune.runner.utils.data_utils import harden_contamination_sets
            harm_dataset = harden_contamination_sets(self.tok, n=256)[1]
        if safety_dataset is None:
            from safetune.runner.utils.data_utils import build_safety_dataset
            safety_dataset = build_safety_dataset(self.tok)

        # with_format (non-mutating) — set_format would mutate the caller's datasets.
        train_dataset, harm_dataset, safety_dataset = (
            ds.with_format("torch") if hasattr(ds, "with_format") else ds
            for ds in (train_dataset, harm_dataset, safety_dataset)
        )
                
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        cfg = HARD.TARConfig(inner_steps=self.inner_steps, inner_lr=self.inner_lr,
                             lambda_tar=self.lambda_tar)
        retain_loader = DataLoader(
            _keep_model_columns(train_dataset), batch_size=self.batch_size, shuffle=True,
            collate_fn=default_data_collator)
        harm_loader = itertools.cycle(DataLoader(
            _keep_model_columns(harm_dataset), batch_size=self.batch_size, shuffle=True,
            collate_fn=default_data_collator))
        safety_loader = itertools.cycle(DataLoader(
            _keep_model_columns(safety_dataset), batch_size=self.batch_size, shuffle=True,
            collate_fn=default_data_collator))
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
        task_loss_fn = lambda m, b: m(**b).loss
        model.train()
        step = 0
        for _ in range(self.epochs):
            for retain_batch in tqdm(retain_loader):
                retain_batch = _to_dev(retain_batch, model.device)
                harm_batch = _to_dev(next(harm_loader), model.device)
                safety_batch = _to_dev(next(safety_loader), model.device)
                loss = HARD.tar_outer_loss(model, retain_batch, harm_batch,
                                           safety_batch, task_loss_fn, cfg)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
        return self._save_merged(model, out_dir)

# ── SaLoRATrainer ─────────────────────────────────────────────────────────────

class SaLoRATrainer(_HardenBase):
    """SaLoRA: safety subspace LoRA projection.

    Args:
        rank: LoRA rank. Default 16.
        safety_rank: safety subspace rank for SVD. Default 16.
        strength: safety projection strength. Default 1.0.
        task_init: initialize LoRA with task direction. Default True.
        n_iter: number of SVD power iterations. Default 7.
        lora_alpha: LoRA alpha (scaling). Defaults to 2 * rank.
        lora_dropout: LoRA dropout rate. Default 0.05.
        target_modules: list of module names to apply LoRA to. Defaults to the
            standard attention + MLP projection set.
    """

    METHOD = "SaLoRA"

    def __init__(self, model=None, tokenizer=None, *,
                 rank: int = 16,
                 safety_rank: int = 16,
                 strength: float = 1.0,
                 task_init: bool = True,
                 n_iter: int = 7,
                 lora_alpha: int = None,
                 lora_dropout: float = 0.05,
                 target_modules: list = None,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.rank = rank
        self.safety_rank = safety_rank
        self.strength = strength
        self.task_init = task_init
        self.n_iter = n_iter
        self.lora_alpha = lora_alpha if lora_alpha is not None else rank * 2
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    def train(self, train_dataset, out_dir: str = None, *,
              safety_dataset=None, **kwargs) -> str:
        out_dir = self._resolve_out_dir(out_dir)
        from peft import LoraConfig, get_peft_model
        from safetune.runner.utils.model_utils import load_model
        # Use the caller's model. (Previously this reloaded base_name from the
        # hub whenever the tokenizer had a name_or_path — silently discarding a
        # user-passed in-memory/drifted model and training a pristine reload.)
        model = self.model
        if model is None:
            base_name = getattr(self.tok, "name_or_path", None)
            model = load_model(base_name)
        model.config.use_cache = False
        lora_cfg = LoraConfig(
            r=self.rank, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_cfg)
        if safety_dataset is None:
            from safetune.runner.utils.data_utils import build_safety_dataset
            safety_dataset = build_safety_dataset(self.tok)
        safety_inputs = [
            # as_tensor + view handles both raw-list and already-tokenized (torch)
            # datasets; torch.tensor([tensor]) raised "only integer tensors of a
            # single element can be converted to an index" on torch-formatted rows.
            {"input_ids": torch.as_tensor(ex["input_ids"], dtype=torch.long).view(1, -1),
             "attention_mask": torch.as_tensor(ex["attention_mask"], dtype=torch.long).view(1, -1)}
            for ex in _keep_model_columns(safety_dataset)
        ]
        cfg = HARD.SaLoRAConfig(rank=self.rank, safety_rank=self.safety_rank,
                                strength=self.strength, task_init=self.task_init)
        subspace = HARD.compute_safety_subspace(model, safety_inputs=safety_inputs,
                                                safety_rank=self.safety_rank,
                                                n_iter=self.n_iter)
        HARD.project_lora_step(model, subspace, cfg, install_forward=True,
                               apply_task_init=self.task_init)
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
            
        tr = Trainer(model=model,
                     args=self._training_args(out_dir),
                     train_dataset=_keep_model_columns(train_dataset),
                     data_collator=default_data_collator)
        tr.train()
        return self._save_merged(model, out_dir)

# ── SEALTrainer ───────────────────────────────────────────────────────────────

class SEALTrainer(_HardenBase):
    """SEAL: data-selection defense (safety-aware example weighting)."""

    METHOD = "SEAL"

    def train(self, train_dataset, out_dir: str = None, *,
              safety_dataset=None, **kwargs) -> str:
        if safety_dataset is None:
            from safetune.runner.utils.data_utils import build_safety_dataset
            safety_dataset = build_safety_dataset(self.tok)
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        args = self._configure_args(HARD.SEALConfig(), out_dir)
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        tr = HARD.SEALTrainer(
            model=model, args=args,
            train_dataset=_keep_model_columns(train_dataset),
            data_collator=default_data_collator,
            safety_dataset=_keep_model_columns(safety_dataset))
        tr.train()
        return self._save_merged(model, out_dir)

# ── ConstrainedSFTTrainer ─────────────────────────────────────────────────────

class ConstrainedSFTTrainer(_HardenBase):
    """ConstrainedSFT: first-token safety constraint during fine-tuning."""

    METHOD = "ConstrainedSFT"

    def train(self, train_dataset, out_dir: str = None, **kwargs) -> str:
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        args = self._configure_args(HARD.ConstrainedSFTConfig(), out_dir)
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        tr = HARD.ConstrainedSFTTrainer(
            model=model, args=args,
            train_dataset=_keep_model_columns(train_dataset),
            data_collator=default_data_collator)
        tr.train()
        return self._save_merged(model, out_dir)

# ── LoXHardenTrainer ─────────────────────────────────────────────────────────

class LoXHardenTrainer(_HardenBase):
    """LoX-Harden: pre-FT low-rank subspace extrapolation for safety.

    Args:
        rank: LoX rank. Default 8.
        extrapolation_factor: extrapolation strength. Default 0.3.
        aligned_model_path: HF path/ID for the aligned reference model.
    """

    METHOD = "LoXHarden"

    def __init__(self, model=None, tokenizer=None, *,
                 rank: int = 8,
                 extrapolation_factor: float = 0.3,
                 aligned_model_path: str = None,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.rank = rank
        self.extrapolation_factor = extrapolation_factor
        self.aligned_model_path = aligned_model_path

    def train(self, train_dataset, out_dir: str = None, **kwargs) -> str:
        out_dir = self._resolve_out_dir(out_dir)
        from safetune.runner.utils.model_utils import load_model_cpu
        if not self.aligned_model_path:
            import logging
            logging.getLogger(__name__).warning(
                "LoXHardenTrainer: no aligned_model_path given — the aligned "
                "reference defaults to the training model, so every weight delta "
                "is zero and LoX is a no-op before plain SFT. Pass "
                "aligned_model_path=<pre-drift aligned checkpoint>."
            )
        aligned_path = (self.aligned_model_path
                        or getattr(self.tok, "name_or_path", None))
        aligned_model = load_model_cpu(aligned_path)
        cfg = HARD.LoXHardenConfig(rank=self.rank,
                                   alpha=self.extrapolation_factor)
        device = next(self.model.parameters()).device
        base_sd = self.model.state_dict()
        aligned_sd = {k: v.to(device) for k, v in aligned_model.state_dict().items()}
        model = HARD.apply_lox_harden(self.model, base_sd, aligned_sd, cfg)
        del aligned_model; free()
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        tr = Trainer(model=model,
                     args=self._training_args(out_dir),
                     train_dataset=_keep_model_columns(train_dataset),
                     data_collator=default_data_collator)
        tr.train()
        return self._save_merged(model, out_dir)

# ── Dataset loader ────────────────────────────────────────────────────────────

def load_harden_data(model_id, n: int = 256, max_len: int = 256):
    """Contaminated SFT train set + refusal safety set. No tokenizer required."""
    from safetune.runner.utils.dataset import load_harden_dataset
    from safetune.runner.utils.model_utils import load_tok
    return load_harden_dataset(load_tok(model_id), n=n, max_len=max_len)

# ── Convenience aliases (all exportable) ─────────────────────────────────────

__all__ = [
    "PlainSFTTrainer",
    "SafeGradTrainer",
    "LisaTrainer",
    "SPPFTTrainer",
    "LookAheadTrainer",
    "STARDSSTrainer",
    "SAPTrainer",
    "AsFTTrainer",
    "SurgeryTrainer",
    "DeRTaTrainer",
    "DOORTrainer",
    "TARTrainer",
    "SaLoRATrainer",
    "VaccineTrainer",
    "TVaccineTrainer",
    "BoosterTrainer",
    "RepNoiseTrainer",
    "CTRAPTrainer",
    "SEAMTrainer",
    "SEALTrainer",
    "ConstrainedSFTTrainer",
    "LoXHardenTrainer",
    "load_harden_data",
]

