"""
AsFT Trainer adapter.

Faithful adapter for "AsFT: Anchoring Safety During LLM Fine-Tuning Within
Narrow Safety Basin" (Yang et al., arXiv:2506.08473, NeurIPS 2025).
Reference implementation: https://github.com/PKU-YuanGroup/AsFT
(``utils/AsFT_train_utils.py`` ``train`` + ``AsFT_finetuning.py`` ``SafeLoRA``).

The authors' algorithm
----------------------
1. For every LoRA-targeted weight matrix, take the alignment direction
   ``d = W_aligned - W_base`` and build a *per-matrix projection matrix*
   ``C = (d @ d.T) / ||d||``  (``SafeLoRA.get_aligned_matrix``).
2. During training the LoRA update of that matrix is ``dW = lora_B @ lora_A``.
   The component of ``dW`` orthogonal to the alignment subspace is
   ``(I - C) @ dW`` and AsFT adds a *loss-level* regularizer

       reg_loss = lambda_reg * sum_matrices || (I - C) @ dW ||_F ** 2

   to the task loss *before* ``backward()`` -- a Lagrangian relaxation that
   penalises updates leaving the narrow safety basin.  The authors' scripts
   all use ``lambda_reg = 1``.

This adapter reproduces (1) and (2) exactly inside ``compute_loss`` so the
penalty enters autograd and shapes the whole optimisation, rather than doing
post-hoc gradient surgery.  When the model has no LoRA modules (full
fine-tuning) the projection-matrix form is not applicable; in that case the
adapter falls back to the equivalent dense form on the full weight delta
``theta - theta_aligned`` via the core ``AsFTWrapper``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from transformers import Trainer, TrainingArguments
    _TRAINER_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    Trainer = object  # type: ignore[assignment,misc]
    TrainingArguments = object  # type: ignore[assignment,misc]
    _TRAINER_IMPORT_ERROR = _e

try:
    from safetune.core.optim.asft import AsFTConfig as _CoreAsFTConfig, AsFTWrapper
    _ASFT_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    AsFTWrapper = None  # type: ignore[assignment]
    _CoreAsFTConfig = None  # type: ignore[assignment]
    _ASFT_IMPORT_ERROR = _e


if _TRAINER_IMPORT_ERROR is None:
    @dataclass
    class AsFTConfig(TrainingArguments):  # type: ignore[misc]
        """TrainingArguments with AsFT safety-basin regularisation hyperparameters."""

        # Lagrangian weight on the orthogonal-component penalty. The AsFT repo's
        # scripts all use 1.0 (``--lambda_reg 1``); keep that as the default.
        reg_lambda: float = 1.0
        # Hard-constraint variant: drop the penalty and instead fully suppress the
        # orthogonal gradient component (only used by the no-LoRA fallback).
        hard_constraint: bool = False
else:  # pragma: no cover
    class AsFTConfig(object):  # type: ignore[assignment]
        pass


class AsFTTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """Trainer adding AsFT's safety-basin loss regularizer to the task loss.

    Faithful path (LoRA models): a per-matrix projection ``C = d d^T / ||d||``
    is built from the aligned-vs-base weight difference of every LoRA-targeted
    matrix; each step the penalty ``reg_lambda * sum ||(I - C) @ (B@A)||_F^2``
    is added inside :meth:`compute_loss` so it enters autograd.

    Fallback path (no LoRA modules): the dense core :class:`AsFTWrapper` is
    used -- ``compute_subspace_penalty`` adds the orthogonal-component penalty
    on the full weight delta, or with ``hard_constraint`` the orthogonal
    gradient component is suppressed post-backward.
    """

    def __init__(
        self,
        *args: Any,
        aligned_state_dict: Optional[Dict[str, Any]] = None,
        base_state_dict: Optional[Dict[str, Any]] = None,
        reg_lambda: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for AsFTTrainer"
            ) from _TRAINER_IMPORT_ERROR
        if _ASFT_IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.optim.asft is unavailable"
            ) from _ASFT_IMPORT_ERROR

        super().__init__(*args, **kwargs)

        self._reg_lambda: float = (
            reg_lambda if reg_lambda is not None
            else getattr(self.args, "reg_lambda", 1.0)
        )
        self._hard_constraint: bool = bool(
            getattr(self.args, "hard_constraint", False)
        )

        # Precomputed (I - C) matrices, keyed by the lora_A parameter name. 
        # Built lazily on first compute_loss call so the PEFT model is available.
        self._proj_matrices: Optional[Dict[str, Any]] = None
        self._aligned_sd = aligned_state_dict
        self._base_sd = base_state_dict
        self._asft_ready = (
            aligned_state_dict is not None and base_state_dict is not None
        )

        # Dense fallback wrapper (only built if the model has no LoRA modules).
        self._asft: Optional[AsFTWrapper] = None
        self._use_lora_path: Optional[bool] = None

    # ------------------------------------------------------------------ #
    # Projection-matrix construction (AsFT_finetuning.SafeLoRA)
    # ------------------------------------------------------------------ #
    def _module_base_name(self, lora_param_name: str) -> Optional[str]:
        """Map a ``...lora_A...`` parameter name to its underlying weight name.

        e.g. ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight``
        -> ``model.layers.0.self_attn.q_proj.weight`` so it can be looked up in
        an aligned/base state dict produced from the un-adapted model.
        """
        name = lora_param_name
        # Strip the PEFT lora_A sub-path, keep up to (and including) the
        # underlying linear module, then re-append ``.weight``.
        marker = ".lora_A"
        idx = name.find(marker)
        if idx == -1:
            return None
        module_path = name[:idx]
        # PEFT prefixes the wrapped model with "base_model.model."; the plain
        # state dict typically does not. Try the stripped form first.
        candidates = [module_path + ".weight"]
        if module_path.startswith("base_model.model."):
            candidates.append(module_path[len("base_model.model."):] + ".weight")
        return candidates  # type: ignore[return-value]

    def _build_proj_matrices(self, model: Any) -> Dict[str, Any]:
        """Build (I - C) where C = d d^T / ||d|| for every LoRA-targeted matrix.

        Mirrors ``SafeLoRA.get_aligned_matrix``: d = W_aligned - W_base for the
        matrix wrapped by each LoRA adapter. We precompute (I - C) for efficiency.
        """
        import torch

        proj: Dict[str, Any] = {}
        if not self._asft_ready:
            return proj
        assert self._aligned_sd is not None and self._base_sd is not None

        named = dict(model.named_parameters())
        for name in named:
            if ".lora_A" not in name:
                continue
            candidates = self._module_base_name(name)
            if not candidates:
                continue
            weight_name = next(
                (c for c in candidates
                 if c in self._aligned_sd and c in self._base_sd),
                None,
            )
            if weight_name is None:
                continue
            
            # Cast to float32 for stable norm calculation
            d = (self._aligned_sd[weight_name].float()
                 - self._base_sd[weight_name].float())
            
            if d.dim() != 2:
                continue
            norm = d.norm()
            if norm.item() <= 0:
                continue
            
            # C = (d @ d.T) / ||d||  -- AsFT_finetuning.py:86
            c = torch.mm(d, d.t()) / norm
            identity = torch.eye(c.shape[0], device=c.device, dtype=c.dtype)
            
            # Precompute and store (I - C) to save training steps
            proj[name] = (identity - c).detach()
            
        return proj

    # ------------------------------------------------------------------ #
    # Loss-level AsFT regularizer (AsFT_train_utils.train, lines 101-117)
    # ------------------------------------------------------------------ #
    def _asft_reg_loss(self, model: Any) -> Any:
        """reg_lambda * sum_matrices || (I - C) @ (lora_B @ lora_A) ||_F ** 2."""
        import torch

        named = dict(model.named_parameters())
        reg: Any = None
        for a_name, a_param in named.items():
            if ".lora_A" not in a_name:
                continue
            if self._proj_matrices is None or a_name not in self._proj_matrices:
                continue
            b_name = a_name.replace("lora_A", "lora_B")
            b_param = named.get(b_name)
            if b_param is None:
                continue
            
            # dW = lora_B @ lora_A   -- AsFT_train_utils.py:110
            delta_w = torch.mm(b_param, a_param)
            
            # Retrieve the precomputed (I - C) matrix
            i_minus_c = self._proj_matrices[a_name].to(
                device=delta_w.device, dtype=delta_w.dtype
            )
            
            # orthogonal component = (I - C) @ dW
            orthogonal = torch.mm(i_minus_c, delta_w)
            term = torch.norm(orthogonal, p="fro") ** 2
            reg = term if reg is None else reg + term
            
        if reg is None:
            return None
        return self._reg_lambda * reg

    def _ensure_setup(self, model: Any) -> None:
        """Decide LoRA vs dense fallback path on first use."""
        if self._use_lora_path is not None:
            return
        if not self._asft_ready:
            # No checkpoints supplied -> AsFT is a no-op (documented behaviour).
            self._use_lora_path = False
            return
        
        has_lora = any(".lora_A" in n for n, _ in model.named_parameters())
        if has_lora:
            self._proj_matrices = self._build_proj_matrices(model)
            self._use_lora_path = bool(self._proj_matrices)
        else:
            self._use_lora_path = False

        if not self._use_lora_path:
            # Dense full-fine-tuning fallback via the core wrapper.
            core_cfg = _CoreAsFTConfig(
                reg_lambda=self._reg_lambda,
                hard_constraint=self._hard_constraint,
            )
            try:
                self._asft = AsFTWrapper(
                    model=model,
                    aligned_state_dict=self._aligned_sd,
                    base_state_dict=self._base_sd,
                    config=core_cfg,
                )
            except Exception as e:
                logger.warning(
                    f"AsFTWrapper initialization failed for dense fallback: {e}. "
                    "The model will fine-tune WITHOUT the safety regularizer."
                )
                self._asft = None

    # ------------------------------------------------------------------ #
    # Trainer hooks
    # ------------------------------------------------------------------ #
    def compute_loss(  # type: ignore[override]
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        """Add the AsFT safety-basin regularizer to the task loss.

        This is the faithful AsFT path: the penalty enters autograd, exactly
        like ``loss = loss + reg_loss`` in the authors' training loop.
        """
        try:
            out = super().compute_loss(
                model, inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        except TypeError:
            # Older transformers without num_items_in_batch
            out = super().compute_loss(
                model, inputs, return_outputs=return_outputs
            )

        if return_outputs:
            loss, outputs = out
        else:
            loss, outputs = out, None

        self._ensure_setup(model)

        reg = None
        if self._use_lora_path:
            reg = self._asft_reg_loss(model)
        elif self._asft is not None and not self._hard_constraint:
            # Dense fallback: orthogonal-component penalty on the full delta.
            try:
                reg = self._asft.compute_subspace_penalty()
            except Exception as e:
                logger.warning(f"Failed to compute dense subspace penalty: {e}")
                reg = None

        if reg is not None:
            loss = loss + reg.to(loss.device, loss.dtype)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):  # type: ignore[override]
        """Run the standard step; apply the hard-constraint surgery if enabled.

        The faithful AsFT regularizer is applied in :meth:`compute_loss`. The
        only post-backward work here is the optional ``hard_constraint`` variant
        for the dense (no-LoRA) fallback, which fully suppresses the orthogonal
        gradient component instead of penalising it.
        """
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
        except TypeError:
            loss = super().training_step(model, inputs)

        self._ensure_setup(model)
        if (
            self._asft is not None
            and not self._use_lora_path
            and self._hard_constraint
        ):
            with self._asft.apply_subspace_constraint():
                pass  # context-manager exit suppresses the orthogonal grad

        return loss