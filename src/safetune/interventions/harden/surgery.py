"""Surgery Trainer adapter: attention-sink *divergence* regularizer.

Implements the fine-tuning-stage defense from "Surgery: Mitigating Harmful
Fine-Tuning for Large Language Models via Attention Sink" (arXiv:2602.05228).

The paper defines a per-head sink value ``alpha_h(X)`` (Eq. 2) and the *sink
divergence* ``d_h = alpha_h(X_m) - alpha_h(X_r)`` (Eq. 3), the difference in
sink value between *harmful* data ``X_m`` and *refusal* data ``X_r``. The
training regularizer (Eq. 4) is ``lambda * (1/|H|) * sum_h ReLU(d_h)`` --
penalising only heads whose harmful-vs-refusal sink gap is positive.

Per the paper's Algorithm 1, a batch of (simulated) refusal data is sampled at
*every* training step and its sink profile becomes the reference ``alpha_h(X_r)``;
a batch of *simulated harmful* data is also sampled to provide ``alpha_h(X_m)``.
This adapter wires that lifecycle in: a ``refusal_dataset`` and a
``harmful_dataset`` are each cycled; a forward pass on a refusal batch sets the
reference profile via ``SurgeryWrapper.set_reference``; a forward pass on the
harmful batch sets the penalty profile ``alpha_h(X_m)``; and
``compute_sink_penalty`` then penalises ``ReLU(curr_sink - ref_sink)``.

**Faithful usage** (matches paper Algorithm 1): supply both ``refusal_dataset``
and ``harmful_dataset`` as separate iterables. The ``inputs`` batch is then used
only for the task cross-entropy loss, not for the sink profile.

**Shortcut** (when user FT data is itself harmful): omit ``harmful_dataset``.
The trainer then uses ``inputs`` as ``X_m``. This is correct when the goal is
to defend against harmful-content fine-tuning attacks, but diverges from the
paper for benign fine-tuning scenarios.

If ``refusal_dataset`` is not supplied the reference cannot be computed; the
trainer falls back to plain training (no penalty).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional

logger = logging.getLogger(__name__)

try:
    from transformers import Trainer, TrainingArguments
    _TRAINER_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    Trainer = object  # type: ignore[assignment,misc]
    TrainingArguments = object  # type: ignore[assignment,misc]
    _TRAINER_IMPORT_ERROR = _e

try:
    from safetune.core.optim.surgery import (
        SurgeryConfig as _CoreSurgeryConfig,
        SurgeryWrapper,
    )
    _SURGERY_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    _CoreSurgeryConfig = None  # type: ignore[assignment]
    SurgeryWrapper = None  # type: ignore[assignment]
    _SURGERY_IMPORT_ERROR = _e


if _TRAINER_IMPORT_ERROR is None:
    @dataclass
    class SurgeryConfig(TrainingArguments):  # type: ignore[misc]
        sink_lambda: float = 0.01
else:  # pragma: no cover
    class SurgeryConfig(object):  # type: ignore[assignment]
        pass


class SurgeryTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """Trainer that penalises harmful-vs-refusal attention-sink divergence.

    Args:
        refusal_dataset: iterable of batches of *refusal* data (``X_r`` in the
            paper, Algorithm 1). Cycled indefinitely; one batch per step sets
            the reference sink profile ``alpha_h(X_r)``. Required for the
            penalty to fire.
        harmful_dataset: optional iterable of batches of *simulated harmful*
            data (``X_m`` in the paper, Algorithm 1 — the separate "harmful
            dataset" required by the paper). When supplied, a batch from this
            iterable is used to measure ``alpha_h(X_m)`` at each step, and
            ``inputs`` provides only the cross-entropy loss. When ``None``
            (default shortcut), ``inputs`` is used as both the CE-loss batch
            and the ``X_m`` sink profile — correct when the user fine-tuning
            data is itself harmful, but diverges from the paper for benign FT.
        **kwargs: forwarded to :class:`transformers.Trainer`.
    """

    def __init__(
        self,
        *args: Any,
        refusal_dataset: Optional[Iterable] = None,
        harmful_dataset: Optional[Iterable] = None,
        **kwargs: Any,
    ) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for SurgeryTrainer"
            ) from _TRAINER_IMPORT_ERROR
        if _SURGERY_IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.optim.surgery is unavailable"
            ) from _SURGERY_IMPORT_ERROR
        
        super().__init__(*args, **kwargs)
        
        sink_lambda = getattr(self.args, "sink_lambda", 0.01)
        self._surgery = SurgeryWrapper(_CoreSurgeryConfig(sink_lambda=sink_lambda))
        
        self._refusal_dataset = refusal_dataset
        self._refusal_iter: Optional[Iterator] = self._build_cyclic_iter(refusal_dataset)
        
        self._harmful_dataset = harmful_dataset
        self._harmful_iter: Optional[Iterator] = self._build_cyclic_iter(harmful_dataset)

    def _build_cyclic_iter(self, dataset: Optional[Iterable]) -> Optional[Iterator]:
        """Creates an infinite iterator over a dataset without memory caching."""
        if dataset is None:
            return None
        def _generator():
            while True:
                for batch in dataset:
                    yield batch
        return _generator()

    def _next_batch(self, iterator: Optional[Iterator], name: str) -> Optional[Any]:
        if iterator is None:
            return None
        try:
            return next(iterator)
        except Exception as e:
            logger.warning(f"Failed to fetch next batch from {name} dataset: {e}")
            return None

    def _prepare_custom_batch(self, batch: Any, model: Any) -> Any:
        """Helper to process inputs efficiently."""
        if hasattr(self, "_prepare_inputs"):
            batch = self._prepare_inputs(batch)
        if isinstance(batch, dict):
            return {
                k: v.to(model.device) if hasattr(v, "to") else v
                for k, v in batch.items()
            }
        return batch

    def _update_reference(self, model) -> bool:
        """Run a forward pass on a refusal batch and set alpha_h(X_r).

        Returns True if the reference sink profile was (re)computed.
        """
        refusal_batch = self._next_batch(self._refusal_iter, "refusal")
        if refusal_batch is None:
            return False
            
        try:
            call_inputs = self._prepare_custom_batch(refusal_batch, model)
            if isinstance(call_inputs, dict):
                call_inputs["output_attentions"] = True
                
            # No grad: the refusal sink profile is a detached reference target.
            import torch

            with torch.no_grad():
                ref_outputs = model(**call_inputs)
                
            ref_attentions = getattr(ref_outputs, "attentions", None)
            if ref_attentions is None:
                if not getattr(self, "_warned_no_attn", False):
                    logger.warning(
                        "Surgery: model returned no attention weights "
                        "(output_attentions is ignored under sdpa/flash "
                        "attention) — the attention-sink penalty is INACTIVE and "
                        "training is plain SFT. Load the model with "
                        "attn_implementation=\"eager\" to enable it."
                    )
                    self._warned_no_attn = True
                return False
                
            self._surgery.set_reference(ref_attentions)
            return True
            
        except Exception as e:
            # A failure to compute the reference must not abort training
            logger.debug(f"Skipping reference update this step due to error: {e}")
            return False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        import torch

        # Step 1: compute the refusal reference sink profile alpha_h(X_r) for
        # this step from a freshly sampled refusal batch (paper Algorithm 1).
        have_reference = self._update_reference(model)

        # Step 2: forward the user batch for the task cross-entropy loss.
        call_inputs = dict(inputs)
        call_inputs["output_attentions"] = True
        outputs = model(**call_inputs)
        base_loss = outputs.loss

        # Step 3: compute alpha_h(X_m) — either from a dedicated harmful batch
        # (paper-faithful path) or from the user inputs (shortcut path).
        harmful_batch = self._next_batch(self._harmful_iter, "harmful")
        
        if harmful_batch is not None:
            # Faithful path: separate harmful dataset (paper Algorithm 1).
            harmful_call = self._prepare_custom_batch(harmful_batch, model)
            if isinstance(harmful_call, dict):
                harmful_call["output_attentions"] = True
                
            # CRITICAL FIX: Removed `torch.no_grad()`. 
            # The penalty requires gradients flowing back through X_m to shape the weights.
            harmful_out = model(**harmful_call)
            attentions = getattr(harmful_out, "attentions", None)
        else:
            # Shortcut: use the user/FT batch as X_m (correct when user data
            # is harmful; diverges from paper for benign FT scenarios).
            attentions = getattr(outputs, "attentions", None)

        # Step 4: penalise the *divergence* d_h = alpha_h(X_m) - alpha_h(X_r),
        # only when a refusal reference was actually computed. Without a
        # reference the penalty would degenerate to raw sink magnitude, so we
        # skip it entirely instead.
        if attentions is not None and have_reference:
            penalty = self._surgery.compute_sink_penalty(attentions)
            if penalty.device != base_loss.device:
                penalty = penalty.to(base_loss.device)
            loss = base_loss + penalty.to(base_loss.dtype)
        else:
            loss = base_loss
            
        return (loss, outputs) if return_outputs else loss