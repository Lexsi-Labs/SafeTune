"""
SafeGrad Trainer adapter.

Wrapper around :class:`transformers.Trainer` that applies the SafeGrad
gradient-surgery technique from "Gradient Surgery for Safe LLM Fine-Tuning"
(Yi et al., arXiv:2508.07172).

Faithfulness notes
------------------
The paper (Algorithm 1) is explicit on two points that an earlier, simplified
implementation diverged from; both are restored here:

1. **Global gradient surgery.** The conflict test ``g_user . g_align < 0`` and
   the orthogonal projection (Eq. 3-4) are applied **once per step on the
   global gradient vector** -- the concatenation of every trainable
   parameter's gradient -- not independently per parameter tensor. This
   trainer therefore flattens all gradients into a single vector, runs one
   conflict test, projects, and scatters the result back. (The per-parameter
   hooks in :class:`SafeGradProjector` are no longer used by this trainer.)

2. **KL-divergence alignment signal + combined update.** The paper's headline
   contribution is that the alignment gradient ``g_align`` is produced by a
   KL-divergence loss ``D_KL(P_{theta_0} || P_theta)`` against a *frozen
   aligned reference model*, and the final update is
   ``g_final = g'_user + rho * g_align`` (Algorithm 1, line ~10-12). This
   trainer wires that in: pass ``reference_model`` to get the paper's KL mode;
   ``rho`` controls the trade-off weight (paper default 1.0).

If neither ``reference_model`` nor ``safety_dataset`` is supplied the trainer
degrades gracefully to vanilla training (with a warning).
"""

from __future__ import annotations

import logging
from itertools import cycle
from typing import Any, Iterable, Iterator, List, Optional

logger = logging.getLogger(__name__)

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
    from safetune.core.optim.safegrad import (
        SafeGradProjector,
        safegrad_kl_alignment_loss,
    )
    _SAFEGRAD_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    SafeGradProjector = None  # type: ignore[assignment]
    safegrad_kl_alignment_loss = None  # type: ignore[assignment]
    _SAFEGRAD_IMPORT_ERROR = _e


class SafeGradConfig(TrainingArguments if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """``TrainingArguments`` subclass for SafeGrad — no extra fields required.

    NOTE: this is intentionally a *plain* subclass, not an ``@dataclass``-
    decorated one. Re-applying ``@dataclass`` to a subclass of
    ``TrainingArguments`` regenerates ``__init__`` from the parent's fields,
    and that regeneration silently produces a zero-field ``__init__`` when
    ``transformers`` was imported before SafeTune. A plain subclass simply
    inherits the working ``TrainingArguments.__init__``.
    """

    pass


class SafeGradTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """Trainer subclass that performs SafeGrad gradient surgery per-step.

    Faithful to "Gradient Surgery for Safe LLM Fine-Tuning" (arXiv:2508.07172):
    a single *global* conflict test + orthogonal projection of the user-task
    gradient against the alignment gradient, and a combined update
    ``g_final = g'_user + rho * g_align``.

    Args:
        safety_dataset: an iterable of batches that can be passed to
            ``model(**batch).loss``. Cycled indefinitely. Used to compute the
            alignment gradient ``g_align``. In *KL mode* (see ``reference_model``)
            these batches supply the alignment data ``D_align`` over which the
            KL divergence to the frozen reference is measured; in *SFT mode*
            (no reference) the cross-entropy loss on these batches is used as a
            proxy alignment signal.
        reference_model: optional frozen, pre-aligned reference model
            (``theta_0`` in the paper). When supplied, ``g_align`` is computed
            from the paper's KL-divergence alignment loss
            ``D_KL(P_reference || P_finetuned)`` on the alignment batch -- the
            paper's headline mode. When ``None``, the SFT-loss proxy is used.
        rho: trade-off weight on the alignment gradient in the combined update
            ``g_final = g'_user + rho * g_align`` (paper default 1.0).
        kl_temperature: softmax temperature for the KL alignment loss
            (paper default 1.0).
        **kwargs: forwarded to :class:`transformers.Trainer`.
    """

    def __init__(
        self,
        *args: Any,
        safety_dataset: Optional[Iterable] = None,
        reference_model: Optional[Any] = None,
        rho: float = 1.0,
        kl_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for SafeGradTrainer"
            ) from _TRAINER_IMPORT_ERROR
        if _TORCH_IMPORT_ERROR is not None:
            raise ImportError(
                "torch is required for SafeGradTrainer"
            ) from _TORCH_IMPORT_ERROR
        if _SAFEGRAD_IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.optim.safegrad is unavailable"
            ) from _SAFEGRAD_IMPORT_ERROR

        super().__init__(*args, **kwargs)
        self._safety_dataset = safety_dataset
        self._safety_iter: Optional[Iterator] = (
            cycle(iter(safety_dataset)) if safety_dataset is not None else None
        )
        self._rho = float(rho)
        self._kl_temperature = float(kl_temperature)

        # Frozen, pre-aligned reference model (theta_0). Kept in eval mode with
        # gradients disabled so no graph flows back into it.
        self._reference_model = reference_model
        if self._reference_model is not None:
            self._reference_model.eval()
            for p in self._reference_model.parameters():
                p.requires_grad_(False)

        # Retained for API compatibility / advanced callers; this trainer does
        # its own global surgery and does not rely on the per-parameter hooks.
        self._projector = SafeGradProjector(self.model)

        if safety_dataset is None and reference_model is None:
            logger.warning(
                "SafeGradTrainer: neither safety_dataset nor reference_model "
                "supplied; gradient surgery is disabled and training reduces "
                "to vanilla fine-tuning."
            )

    # ------------------------------------------------------------------ #
    # safety batch plumbing
    # ------------------------------------------------------------------ #
    def _next_safety_batch(self) -> Optional[Any]:
        if self._safety_iter is None:
            return None
        try:
            return next(self._safety_iter)
        except StopIteration:
            if self._safety_dataset is None:
                return None
            self._safety_iter = cycle(iter(self._safety_dataset))
            return next(self._safety_iter)

    def _prepare_batch(self, batch: Any, model: Any) -> Any:
        """Move a batch onto the model's device, using HF's helper if present."""
        if hasattr(self, "_prepare_inputs"):
            batch = self._prepare_inputs(batch)
        if isinstance(batch, dict):
            device = getattr(model, "device", None)
            if device is not None:
                batch = {
                    k: (v.to(device) if hasattr(v, "to") else v)
                    for k, v in batch.items()
                }
        return batch

    # ------------------------------------------------------------------ #
    # global gradient surgery (paper Algorithm 1, Eq. 3-4)
    # ------------------------------------------------------------------ #
    def _trainable_params(self, model: Any) -> List[Any]:
        return [p for p in model.parameters() if p.requires_grad]

    def _alignment_gradient(self, model: Any, safety_batch: Any) -> Optional[List[Any]]:
        """Compute the alignment gradient g_align over the trainable params.

        KL mode (paper headline): if a frozen ``reference_model`` is available,
        run both the fine-tuned model and the reference on the alignment batch
        and backprop ``D_KL(P_reference || P_finetuned)``.

        SFT mode: otherwise backprop the cross-entropy loss on the batch.
        """
        params = self._trainable_params(model)
        if not params:
            return None
        try:
            prepared = self._prepare_batch(safety_batch, model)
            if not isinstance(prepared, dict):
                logger.warning(
                    "SafeGradTrainer: safety batch is not a dict; cannot "
                    "compute alignment gradient this step."
                )
                return None

            if self._reference_model is not None and safegrad_kl_alignment_loss is not None:
                # Paper's KL-divergence alignment loss against frozen theta_0.
                student_out = model(**prepared)
                with torch.no_grad():
                    ref_inputs = {
                        k: (v.to(self._reference_model.device)
                            if hasattr(v, "to") else v)
                        for k, v in prepared.items()
                    }
                    ref_out = self._reference_model(**ref_inputs)
                aligned_logits = ref_out.logits.detach().to(student_out.logits.device)
                align_loss = safegrad_kl_alignment_loss(
                    student_out.logits,
                    aligned_logits,
                    temperature=self._kl_temperature,
                )
            else:
                # SFT-loss proxy alignment signal.
                align_loss = model(**prepared).loss

            if align_loss is None or not getattr(align_loss, "requires_grad", False):
                logger.warning(
                    "SafeGradTrainer: alignment loss does not require grad; "
                    "skipping surgery this step."
                )
                return None

            grads = torch.autograd.grad(
                align_loss, params, allow_unused=True, retain_graph=False
            )
            return [
                g.detach() if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, params)
            ]
        except Exception as exc:  # pragma: no cover - robustness guard
            logger.warning(
                "SafeGradTrainer: failed to compute alignment gradient (%s); "
                "skipping surgery this step.",
                exc,
            )
            return None

    def _apply_global_surgery(self, model: Any, g_align: List[Any]) -> None:
        """Project the (already-populated) ``.grad`` of every trainable param.

        Implements the paper's Algorithm 1 on the *global* gradient vector:
        flatten all parameter gradients into a single vector, run one conflict
        test ``g_user . g_align < 0``, project (Eq. 4), then form the combined
        update ``g_final = g'_user + rho * g_align`` and scatter back into
        ``.grad`` so the downstream optimizer steps on it.
        """
        params = self._trainable_params(model)
        if not params:
            return

        flat_user: List[Any] = []
        flat_align: List[Any] = []
        for p, ga in zip(params, g_align):
            gu = p.grad if p.grad is not None else torch.zeros_like(p)
            flat_user.append(gu.reshape(-1))
            flat_align.append(ga.reshape(-1).to(gu.device, gu.dtype))

        g_user_vec = torch.cat(flat_user)
        g_align_vec = torch.cat(flat_align)

        # NOTE: use elementwise-multiply + sum rather than ``torch.dot``.
        # ``torch.dot`` dispatches to a BLAS routine whose length argument is
        # int32, so it overflows ("dot only supports ... bound 2147483647")
        # once the concatenated global gradient exceeds ~2.1B elements — which
        # happens for any model above ~2B parameters. ``(a*b).sum()`` is
        # numerically identical and has no such limit.
        dot_user_align = (g_user_vec * g_align_vec).sum()
        align_sq = (g_align_vec * g_align_vec).sum() + 1e-12

        # Eq. 3-4: project only when the global gradients conflict.
        if dot_user_align < 0:
            g_user_vec = g_user_vec - (dot_user_align / align_sq) * g_align_vec

        # Combined update: g_final = g'_user + rho * g_align (Algorithm 1).
        g_final_vec = g_user_vec + self._rho * g_align_vec

        # Scatter g_final back into each parameter's .grad.
        offset = 0
        for p in params:
            numel = p.numel()
            chunk = g_final_vec[offset:offset + numel].reshape(p.shape)
            if p.grad is None:
                p.grad = chunk.detach().clone()
            else:
                p.grad.copy_(chunk)
            offset += numel

    # ------------------------------------------------------------------ #
    # training step
    # ------------------------------------------------------------------ #
    def training_step(self, model, inputs, num_items_in_batch=None):  # type: ignore[override]
        # 1. User-task forward/backward populates model parameter .grad with
        #    g_user (HF Trainer.training_step does loss.backward() internally).
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
        except TypeError:
            # Older transformers signatures without num_items_in_batch.
            loss = super().training_step(model, inputs)

        # 2. Compute the alignment gradient g_align (KL mode or SFT proxy).
        safety_batch = self._next_safety_batch()
        if safety_batch is None and self._reference_model is None:
            # No alignment signal configured -> vanilla training.
            return loss

        if safety_batch is None:
            # Reference set but no alignment data to evaluate KL on.
            logger.warning(
                "SafeGradTrainer: reference_model supplied without a "
                "safety_dataset; no alignment data this step."
            )
            return loss

        g_align = self._alignment_gradient(model, safety_batch)
        if g_align is None:
            return loss

        # 3. Global conflict test + projection + combined update (Eq. 3-4).
        self._apply_global_surgery(model, g_align)
        return loss
