"""
STAR-DSS Trainer adapter.

Thin wrapper around :class:`transformers.Trainer` that applies the
Dynamic Safety Shaping loss from ``safetune.core.optim.star_dss``.

Reference
---------
"Shape it Up! Restoring LLM Safety during Finetuning" (STAR-DSS),
Peng et al., NeurIPS 2025, arXiv:2505.17196.
Original code: https://github.com/poloclub/star-dss
(``src/model/loss.py::ValueWeightedGPTLMLoss`` and
``src/trainer/sft_trainer.py::SFTTrainer``).

Faithfulness note
-----------------
The defining objective of *⋆DSS* is Eq. (3) of the paper::

    L = Σ_k Σ_t  V_safe(x, y_{1:kM}) · L_CE(y_t)
                 + (1 − V_safe(x, y_{1:kM})) · λ_KL · L_KL

i.e. token-level STAR scores ``V_safe`` interpolate between *imitation*
(cross-entropy on safe segments) and *safety regularization* (KL to a
frozen reference policy ``π_ref`` on unsafe segments). The KL-to-reference
term is **not optional** — it is precisely the mechanism that *suppresses*
unsafe content: as the paper states, "when ∀t, V_safe ≈ 0, the KL term
dominates, nudging the model toward the reference distribution and
discouraging unsafe learning."

The authors' ``SFTTrainer`` therefore holds a frozen ``ref_model`` and
forwards it every step to produce ``ref_logits`` (see ``sft_trainer.py``
lines 276-277 and 398-421). Earlier versions of this adapter never
constructed a reference model and left ``use_kl_penalty`` off by default,
so the suppression term silently never fired and ⋆DSS degraded to a pure
non-negative down-weighting of unsafe tokens. This adapter restores the
reference model and wires the KL suppression term so the loss matches the
paper's Eq. (3).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Optional

try:
    from transformers import Trainer, TrainingArguments
    _TRAINER_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    Trainer = object  # type: ignore[assignment,misc]
    TrainingArguments = object  # type: ignore[assignment,misc]
    _TRAINER_IMPORT_ERROR = _e

try:
    import torch
    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = _e

try:
    from safetune.core.optim.star_dss import DynamicSafetyShapingLoss
    _STAR_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    DynamicSafetyShapingLoss = None  # type: ignore[assignment]
    _STAR_IMPORT_ERROR = _e

logger = logging.getLogger(__name__)


if _TRAINER_IMPORT_ERROR is None:
    @dataclass
    class STARDSSConfig(TrainingArguments):  # type: ignore[misc]
        """TrainingArguments for STAR-DSS dynamic shaping.

        Attributes
        ----------
        use_kl_penalty:
            Enable the KL-to-reference suppression term of Eq. (3). The paper's
            ⋆DSS *always* uses this term — it is what suppresses unsafe content.
            Default is ``True`` so the adapter is faithful out of the box; the
            term still only contributes when a reference model is available.
        kl_scale:
            ``λ_KL`` in Eq. (3): the scaling factor of the KL regularization
            term applied on low-safety (unsafe) tokens.
        """

        use_kl_penalty: bool = True
        kl_scale: float = 1.0
        # STAR-DSS needs the per-token `safety_weights` column at loss time; HF's
        # default (True) would strip it and silently fall back to vanilla SFT for
        # direct-API users. The SafeTune runner already forces this off.
        remove_unused_columns: bool = False
else:  # pragma: no cover
    class STARDSSConfig(object):  # type: ignore[assignment]
        pass


class STARDSSTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """Trainer that applies the STAR-DSS dynamic-safety-shaping loss.

    If a batch contains a ``safety_weights`` tensor (the per-token STAR
    scores ``V_safe``), :class:`DynamicSafetyShapingLoss` computes the
    Eq. (3) objective: ``V_safe`` weighted cross-entropy plus an
    ``(1 − V_safe)·λ_KL`` weighted KL divergence to a frozen reference
    policy. The KL-to-reference term is what *suppresses* unsafe content,
    so a frozen reference model is required for it to take effect.

    Parameters
    ----------
    ref_model:
        Optional frozen reference policy ``π_ref`` (Eq. 3). If ``None`` and
        ``use_kl_penalty`` is enabled, the trainer snapshots a frozen,
        deep-copied clone of the initial ``model`` to serve as ``π_ref`` —
        mirroring the finetuning-as-a-service setting of the paper where
        the aligned base model is the reference. Pass ``ref_model=False``
        to explicitly disable the reference (degrades to pure down-weighting).
    """

    def __init__(
        self,
        *args: Any,
        ref_model: Any = None,
        **kwargs: Any,
    ) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for STARDSSTrainer"
            ) from _TRAINER_IMPORT_ERROR
        if _STAR_IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.optim.star_dss is unavailable"
            ) from _STAR_IMPORT_ERROR

        super().__init__(*args, **kwargs)

        self._use_kl_penalty = bool(getattr(self.args, "use_kl_penalty", True))
        self._kl_scale = float(getattr(self.args, "kl_scale", 1.0))
        self._shaping_loss = DynamicSafetyShapingLoss(
            use_kl_penalty=self._use_kl_penalty,
            kl_scale=self._kl_scale,
        )

        # Reference policy pi_ref for the KL suppression term of Eq. (3).
        # The authors' SFTTrainer keeps a frozen ref_model and forwards it
        # every step (star-dss/src/trainer/sft_trainer.py:276-277,398-421).
        self._ref_model = None
        if self._use_kl_penalty:
            if ref_model is False:
                logger.warning(
                    "STARDSSTrainer: use_kl_penalty is on but ref_model is "
                    "disabled; the unsafe-content suppression (KL) term of "
                    "STAR-DSS Eq. (3) will not fire."
                )
            elif ref_model is not None:
                self._ref_model = self._prepare_ref_model(ref_model)
            else:
                # Snapshot the initial (aligned) model as the reference.
                self._ref_model = self._prepare_ref_model(
                    copy.deepcopy(self.model)
                )

    def _prepare_ref_model(self, ref_model: Any) -> Any:
        """Freeze the reference model and place it on the training device."""
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        try:
            ref_model.to(self.args.device)
        except Exception:  # pragma: no cover - device placement is best-effort
            pass
        return ref_model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        safety_weights = None
        if isinstance(inputs, dict) and "safety_weights" in inputs:
            safety_weights = inputs.pop("safety_weights")

        if safety_weights is None:
            try:
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
                )
            except TypeError:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)

        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits

        # Reference logits for the KL-to-pi_ref suppression term (Eq. 3).
        ref_logits = None
        if self._use_kl_penalty and self._ref_model is not None:
            with torch.no_grad():
                ref_outputs = self._ref_model(**model_inputs)
            ref_logits = ref_outputs.logits

        loss = self._shaping_loss(
            logits=logits,
            labels=labels,
            safety_weights=safety_weights,
            ref_logits=ref_logits,
        )
        return (loss, outputs) if return_outputs else loss
