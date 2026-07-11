"""
Constrained-SFT (Qi et al., ICLR 2025, arXiv:2406.05946).

Key finding from the paper: safety alignment in LLMs is concentrated in the
first few tokens of the response (the "safety critical" prefix). Standard SFT
on benign data erodes this first-token distribution even when the training data
is entirely harmless.

⚠️ FIDELITY: this is a SafeTune VARIANT that captures the paper's central
*insight* (protect the first-token distribution most) but uses a different
*objective form*. The paper's Eq.3 is a per-token bounded term
``(2/β_t)·logσ(β_t·log[p_model/p_ref])`` with a **step-function** β_t schedule
(≈ 0.5 / 2 / 0.1 over the first positions). We instead add an explicit
position-decaying reverse-KL penalty on top of plain SFT with an exponential
β_t schedule (below). Same protective intent, NOT the paper's exact loss.

This variant adds a position-decaying KL penalty to the standard SFT loss:

    L_total = L_SFT + sum_t [ beta_t * KL(p_ref(.|context) || p_model(.|context)) ]

where beta_t = beta * exp(-decay_rate * t) is large for early positions (t=0,1,2)
and decays exponentially, protecting the first-token distribution most strongly.

The reference model p_ref is the aligned model BEFORE fine-tuning (frozen).
The KL direction KL(p_ref || p_model) is the reverse KL — zero-avoiding,
meaning the model is strongly penalised for assigning near-zero probability
to tokens where the aligned reference has significant mass. This is the correct
direction for a safety constraint (prevents the model from forgetting safe
response prefixes).

Paper defaults: beta=0.5, decay_rate=0.1.

Usage::

    from safetune.harden.constrained_sft import ConstrainedSFTTrainer, ConstrainedSFTConfig

    config = ConstrainedSFTConfig(
        output_dir="csft_out",
        csft_beta=0.5,
        csft_decay_rate=0.1,
    )
    trainer = ConstrainedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        reference_model=frozen_aligned_model,
    )
    trainer.train()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from transformers import Trainer, TrainingArguments
    _TRAINER_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    Trainer = object  # type: ignore[assignment,misc]
    TrainingArguments = object  # type: ignore[assignment,misc]
    _TRAINER_IMPORT_ERROR = _e


if _TRAINER_IMPORT_ERROR is None:
    @dataclass
    class ConstrainedSFTConfig(TrainingArguments):  # type: ignore[misc]
        """TrainingArguments subclass for Constrained-SFT.

        Attributes:
            csft_beta: Scale of the KL penalty at position 0.
                beta_t = csft_beta * exp(-csft_decay_rate * t).
                Paper default: 0.5.
            csft_decay_rate: Exponential decay rate over token positions.
                Higher values concentrate the constraint on earlier positions.
                Paper default: 0.1.
        """
        csft_beta: float = 0.5
        csft_decay_rate: float = 0.1
else:  # pragma: no cover
    class ConstrainedSFTConfig(object):  # type: ignore[assignment]
        pass


class ConstrainedSFTTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """HuggingFace Trainer implementing the Constrained-SFT objective.

    Adds a position-decaying KL penalty to the standard SFT cross-entropy loss
    to protect the first-token safety distribution during fine-tuning.

    L_total = L_SFT + sum_t [ beta_t * KL(p_ref || p_model) ] over valid tokens

    where beta_t = beta * exp(-decay_rate * t).

    Args:
        reference_model: A frozen copy of the aligned model (before fine-tuning).
            Must have requires_grad=False on all parameters. If None, the trainer
            falls back to plain SFT (graceful degradation).
        All other arguments forwarded to transformers.Trainer.
    """

    def __init__(
        self,
        *args: Any,
        reference_model: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for ConstrainedSFTTrainer"
            ) from _TRAINER_IMPORT_ERROR

        super().__init__(*args, **kwargs)

        self._ref_model = reference_model
        if reference_model is not None:
            # Ensure the reference is fully frozen.
            for p in reference_model.parameters():
                p.requires_grad_(False)
            reference_model.eval()
        else:
            logger.warning(
                "ConstrainedSFTTrainer: no reference_model supplied — "
                "falling back to plain SFT (KL constraint disabled)."
            )

        self._csft_beta = float(getattr(self.args, "csft_beta", 0.5))
        self._csft_decay_rate = float(getattr(self.args, "csft_decay_rate", 0.1))

    def compute_loss(  # type: ignore[override]
        self,
        model: Any,
        inputs: Any,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Any:
        """Compute L_SFT + position-decaying KL constraint."""

        # ---- Standard SFT loss ----------------------------------------
        outputs = model(**inputs)
        sft_loss = outputs.loss

        if self._ref_model is None:
            return (sft_loss, outputs) if return_outputs else sft_loss

        # ---- Position-decaying KL constraint --------------------------
        labels = inputs.get("labels")
        logits = outputs.logits  # (B, S, V)
        B, S, V = logits.shape

        with torch.no_grad():
            ref_logits = self._ref_model(**inputs).logits  # (B, S, V)

        # ---- Causal shift ---------------------------------------------
        # In a causal LM, the logits at position p predict the token at
        # position p+1.  To constrain the distribution that *predicts* each
        # response token we must align logits[..., :-1, :] with labels[..., 1:]
        # (the standard next-token shift).  Without this shift the logits
        # position predicting the FIRST response token is scored against the
        # (masked) prompt label, so it receives valid=0 and the decay clock is
        # one token late.
        shift_logits = logits[..., :-1, :]         # (B, S-1, V)
        shift_ref    = ref_logits[..., :-1, :]     # (B, S-1, V)
        if labels is not None:
            shift_labels = labels[..., 1:]          # (B, S-1)
        else:
            shift_labels = None
        Sm1 = shift_logits.shape[1]

        # KL(p_ref || p_model) per position — reverse KL, zero-avoiding.
        # = sum_v p_ref * (log p_ref - log p_model)
        log_p_model = F.log_softmax(shift_logits, dim=-1)   # (B, S-1, V)
        log_p_ref   = F.log_softmax(shift_ref, dim=-1)      # (B, S-1, V)
        p_ref       = log_p_ref.exp()                        # (B, S-1, V)

        kl_per_pos = (p_ref * (log_p_ref - log_p_model)).sum(-1)  # (B, S-1)

        # Mask: only apply at logits positions that predict a valid (non-prompt,
        # non-pad) response token, i.e. where the *shifted* label is not -100.
        if shift_labels is not None:
            valid = (shift_labels != -100).float()          # (B, S-1)
        else:
            valid = torch.ones(B, Sm1, device=logits.device)

        # Position weights: beta_t = beta * exp(-decay_rate * t), with t=0 at the
        # logits position that predicts each row's FIRST response token and
        # increasing along the response.  Because the shift is already applied,
        # `first_resp` is the first index where the shifted label is a response
        # token — exactly the position whose logits predict that token.
        abs_pos = torch.arange(Sm1, device=logits.device).unsqueeze(0)  # (1, S-1)
        if shift_labels is not None:
            valid_bool = shift_labels != -100
            # First response index per row; clamp handles all-masked rows.
            first_resp = torch.argmax(valid_bool.int(), dim=1, keepdim=True)  # (B, 1)
        else:
            first_resp = torch.zeros(B, 1, dtype=torch.long, device=logits.device)
        t = (abs_pos - first_resp).clamp(min=0).to(logits.dtype)  # (B, S-1)
        pos_weights = self._csft_beta * torch.exp(-self._csft_decay_rate * t)  # (B, S-1)

        # Weighted sum over valid positions.
        weighted_kl = (kl_per_pos * pos_weights * valid).sum() \
                      / valid.sum().clamp(min=1.0)

        loss = sft_loss + weighted_kl

        return (loss, outputs) if return_outputs else loss


__all__ = ["ConstrainedSFTConfig", "ConstrainedSFTTrainer"]