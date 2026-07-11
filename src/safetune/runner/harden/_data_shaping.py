"""Harden runner — data-shaping trainers (Lisa, SPPFT, LookAhead, STARDSS, DeRTa)."""
from ._base import (
    _HardenBase, _apply_training_args, _keep_model_columns,
    default_data_collator, _derta_collator, _stardss_collator,
)
import safetune.harden as HARD
from torch.utils.data import DataLoader

import os
import torch
from transformers import AutoModelForCausalLM

# ── LisaTrainer ───────────────────────────────────────────────────────────────

class LisaTrainer(_HardenBase):
    """Lisa: bi-state proximal optimization (alignment/finetune alternation).

    Args:
        lisa_rho: proximal constraint weight. Default 0.1.
        lisa_warmup_steps: warm-up steps before alternation starts. Default 10.
        lisa_alignment_step: steps per alignment phase. Default 20.
        lisa_finetune_step: steps per fine-tune phase. Default 20.
    """

    METHOD = "Lisa"

    def __init__(self, model=None, tokenizer=None, *,
                 lisa_rho: float = 0.1,
                 lisa_warmup_steps: int = 10,
                 lisa_alignment_step: int = 20,
                 lisa_finetune_step: int = 20,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.lisa_rho = lisa_rho
        self.lisa_warmup_steps = lisa_warmup_steps
        self.lisa_alignment_step = lisa_alignment_step
        self.lisa_finetune_step = lisa_finetune_step

    def train(self, train_dataset, out_dir: str = None, *,
              safety_dataset=None, **kwargs) -> str:
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        args = HARD.LisaConfig(
            lisa_rho=self.lisa_rho,
            lisa_warmup_steps=self.lisa_warmup_steps,
            lisa_alignment_step=self.lisa_alignment_step,
            lisa_finetune_step=self.lisa_finetune_step,
        )
        self._configure_args(args, out_dir)
        if safety_dataset is None:
            from safetune.runner.utils.data_utils import build_safety_dataset
            safety_dataset = build_safety_dataset(self.tok)

        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        if hasattr(safety_dataset, "with_format"):
            safety_dataset = safety_dataset.with_format("torch")

        alignment_loader = DataLoader(
            _keep_model_columns(safety_dataset), batch_size=self.batch_size, shuffle=True,
            collate_fn=default_data_collator)
        tr = HARD.LisaTrainer(
            model=model, args=args,
            train_dataset=_keep_model_columns(train_dataset),
            data_collator=default_data_collator,
            alignment_dataset=alignment_loader)
        tr.train()
        return self._save_merged(model, out_dir)

# ── SPPFTTrainer ──────────────────────────────────────────────────────────────

class SPPFTTrainer(_HardenBase):
    """SPPFT: self-play prompt fine-tuning."""

    METHOD = "SPPFT"

    def train(self, train_dataset, out_dir: str = None, **kwargs) -> str:
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        # Thread sppft_mode through so it isn't silently ignored (default freeze).
        sppft_mode = self._extra.get("sppft_mode", kwargs.get("sppft_mode", "freeze"))
        args = self._configure_args(HARD.SPPFTConfig(sppft_mode=sppft_mode), out_dir)
        tr = HARD.SPPFTTrainer(
            model=model, args=args,
            train_dataset=_keep_model_columns(train_dataset),
            data_collator=default_data_collator)
        tr.train()
        return self._save_merged(model, out_dir)

# ── LookAheadTrainer ─────────────────────────────────────────────────────────

class LookAheadTrainer(_HardenBase):
    """LookAhead: virtual-token prefix safety lookahead.

    Args:
        prefix_mode: ``"virtual"`` or ``"token"``. Default ``"virtual"``.
        prefix_length: number of prefix tokens. Default 6.
    """

    METHOD = "LookAhead"

    def __init__(self, model=None, tokenizer=None, *,
                 prefix_mode: str = "virtual",
                 prefix_length: int = 6,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.prefix_mode = prefix_mode
        self.prefix_length = prefix_length

    def train(self, train_dataset, out_dir: str = None, **kwargs) -> str:
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        args = self._configure_args(
            HARD.LookAheadConfig(prefix_mode=self.prefix_mode,
                                 prefix_length=self.prefix_length),
            out_dir)
        tr = HARD.LookAheadTrainer(
            model=model, args=args,
            train_dataset=_keep_model_columns(train_dataset),
            data_collator=default_data_collator,
            processing_class=self.tok)
        tr.train()
        return self._save_merged(model, out_dir)

# ── STARDSSTrainer ────────────────────────────────────────────────────────────

class STARDSSTrainer(_HardenBase):
    """STAR-DSS: per-token safety weights with optional KL penalty.

    Args:
        use_kl_penalty: whether to add KL penalty term. Default True.
        kl_scale: KL penalty scale. Default 1.0.
    """

    METHOD = "STARDSS"

    def __init__(self, model=None, tokenizer=None, *,
                 use_kl_penalty: bool = True,
                 kl_scale: float = 1.0,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.use_kl_penalty = use_kl_penalty
        self.kl_scale = kl_scale

    def train(self, train_dataset, out_dir: str = None, **kwargs) -> str:
        from datasets import Dataset
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        args = self._configure_args(
            HARD.STARDSSConfig(use_kl_penalty=self.use_kl_penalty,
                               kl_scale=self.kl_scale),
            out_dir)
        seq_len = len(train_dataset[0]["input_ids"])
        rows = []
        for r in train_dataset:
            v = 1.0 if r.get("kind") == "benign" else 0.0
            rows.append({"input_ids": r["input_ids"],
                         "attention_mask": r["attention_mask"],
                         "labels": r["labels"],
                         "safety_weights": [v] * seq_len})

        stardss_ds = Dataset.from_list(rows)
        stardss_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "safety_weights"])

        tr = HARD.STARDSSTrainer(
            model=model, args=args,
            train_dataset=stardss_ds,
            data_collator=_stardss_collator)
        tr.train()
        return self._save_merged(model, out_dir)

# ── DeRTaTrainer ──────────────────────────────────────────────────────────────

class DeRTaTrainer(_HardenBase):
    """DeRTa: per-token safe/unsafe token classification during training."""

    METHOD = "DeRTa"

    def train(self, train_dataset, out_dir: str = None, *,
              contamination_pairs=None, refusal_pairs=None, **kwargs) -> str:
        from datasets import Dataset
        from safetune.harden.derta import prepare_derta_dataset

        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        args = self._configure_args(HARD.DeRTaConfig(), out_dir)

        if contamination_pairs is None or refusal_pairs is None:
            from safetune.runner.utils.data_utils import harden_contamination_pairs
            _cont, _, _ref = harden_contamination_pairs(256)
            contamination_pairs = _cont
            refusal_pairs = _ref

        raw = [{"prompt": u, "harmful_response": bad, "safe_response": ref}
               for (u, bad), (_, ref) in zip(contamination_pairs, refusal_pairs)]

        def _derta_tokenize(tok, prompt, response, max_len=256):
            prompt_text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
            enc = tok(prompt_text + response, truncation=True, max_length=max_len,
                      padding="max_length")
            plen = len(tok(prompt_text, truncation=True, max_length=max_len)["input_ids"])
            labels = list(enc["input_ids"])
            for i in range(len(labels)):
                if i < plen or enc["attention_mask"][i] == 0:
                    labels[i] = -100
            return enc["input_ids"], enc["attention_mask"], labels

        rows = []
        for r in prepare_derta_dataset(raw):
            ids, mask, labels = _derta_tokenize(self.tok, r["prompt"], r["response"])
            rows.append({"input_ids": ids, "attention_mask": mask,
                         "labels": labels, "safe": bool(r["safe"])})
            
        derta_ds = Dataset.from_list(rows)
        # `safe` MUST be in the formatted columns — otherwise set_format drops it
        # (output_all_columns defaults to False) and _derta_collator falls back to
        # f.get("safe", True), marking every example safe and silently disabling
        # DeRTa's harmful/safe distinction. (cf. STARDSS which keeps safety_weights.)
        derta_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "safe"])

        tr = HARD.DeRTaTrainer(
            model=model, args=args, processing_class=self.tok,
            train_dataset=derta_ds, data_collator=_derta_collator)
        tr.train()
        return self._save_merged(model, out_dir)

