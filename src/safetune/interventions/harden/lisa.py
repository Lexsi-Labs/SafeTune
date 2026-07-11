"""Lisa Trainer adapter.

Thin wrapper around :class:`transformers.Trainer` that applies the Lisa
bi-state proximal optimization from ``safetune.core.optim.lisa``.

Reference: Huang et al., "Lisa: Lazy Safety Alignment for Large Language
Models against Harmful Fine-tuning" (NeurIPS 2024). arXiv:2405.18641.
Repo: https://github.com/git-disl/Lisa
"""
from __future__ import annotations

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
    from safetune.core.optim.lisa import LisaOptimizerWrapper
    _LISA_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    LisaOptimizerWrapper = None  # type: ignore[assignment]
    _LISA_IMPORT_ERROR = _e


if _TRAINER_IMPORT_ERROR is None:
    @dataclass
    class LisaConfig(TrainingArguments):  # type: ignore[misc]
        """TrainingArguments with Lisa Bi-State Optimization hyperparameters.

        The authors' Lisa (`git-disl/Lisa`, ``trainer.py``) alternates between an
        *alignment* state (trained on alignment/safety data) and a *finetune* state
        (trained on the user task data), switching every ``alignment_step`` /
        ``finetune_step`` steps, and adds a proximal L2 term anchoring the active
        state's weights to the *other* state's last consensus checkpoint.

        Attributes:
            lisa_rho: proximal penalty coefficient ``rho`` (paper default 0.1).
            lisa_warmup_steps: steps before the proximal penalty begins firing.
                The authors skip the proximal term for the first ~10% of total
                training steps; this exposes that as an explicit step count.
            lisa_alignment_step: number of consecutive steps spent in the
                *alignment* state before switching to *finetune* (paper default
                500). 0 disables the alignment state entirely.
            lisa_finetune_step: number of consecutive steps spent in the
                *finetune* state before switching back to *alignment* (paper
                default 500). 0 disables bi-state switching.
        """

        lisa_rho: float = 0.1
        lisa_warmup_steps: int = 100
        lisa_alignment_step: int = 500
        lisa_finetune_step: int = 500
else:  # pragma: no cover
    class LisaConfig(object):  # type: ignore[assignment]
        pass


class LisaTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """Trainer implementing Lisa's Bi-State Optimization (BSO) with proximal term.

    Faithful to the authors' ``LisaTrainer`` (``git-disl/Lisa/trainer.py``):

    * **Bi-state dataset alternation.** Two states, ``alignment`` and
      ``finetune``, are trained on *two different datasets*. The ``finetune``
      state consumes the standard ``train_dataset`` (user task data); the
      ``alignment`` state consumes a separate ``alignment_dataset`` (safety
      data). ``training_step`` swaps in an alignment batch whenever the active
      state is ``alignment`` -- this is the defining "lazy alignment" mechanism.
    * **Interval switching.** State flips every ``lisa_alignment_step`` /
      ``lisa_finetune_step`` steps via ``check_mode`` (mirrors the authors'
      ``clock``-based ``check_mode``). On each switch the outgoing state's
      weights are committed to its consensus tracker.
    * **Proximal term.** Injected through the wrapper's
      ``apply_proximal_penalty`` context manager inside ``compute_loss`` so it
      enters the autograd graph before backward. In the ``alignment`` state it
      anchors to the ``finetune`` consensus; in the ``finetune`` state it
      anchors to the ``alignment`` consensus.

    Args:
        alignment_dataset: dataset of safety/alignment examples for the
            alignment state. If ``None`` (or if ``lisa_alignment_step`` is 0),
            the trainer degrades gracefully to single-dataset proximal-only
            training (the pre-fix behaviour) and never enters the alignment
            state.

    The public construction signature is unchanged: ``alignment_dataset`` is an
    optional keyword argument with a default of ``None``.
    """

    def __init__(
        self,
        *args: Any,
        alignment_dataset: Any = None,
        **kwargs: Any,
    ) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for LisaTrainer"
            ) from _TRAINER_IMPORT_ERROR
        if _LISA_IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.optim.lisa is unavailable"
            ) from _LISA_IMPORT_ERROR

        super().__init__(*args, **kwargs)
        self._lisa = LisaOptimizerWrapper(
            model=self.model,
            rho=getattr(self.args, "lisa_rho", 0.1),
            warmup_steps=getattr(self.args, "lisa_warmup_steps", 100),
        )

        # Bi-state switching intervals (authors' alignment_step / finetune_step).
        self._lisa_alignment_step = int(getattr(self.args, "lisa_alignment_step", 500))
        self._lisa_finetune_step = int(getattr(self.args, "lisa_finetune_step", 500))

        # Separate alignment dataset / dataloader for the alignment state.
        self._lisa_alignment_dataset = alignment_dataset
        self._lisa_alignment_loader = None
        self._lisa_alignment_iter = None
        # Bi-state is only active when we have both an alignment dataset and a
        # non-zero alignment interval (mirrors the authors' guard:
        # `alignment_step != 0 and guide_data_num > 0`).
        self._lisa_bistate_enabled = (
            alignment_dataset is not None and self._lisa_alignment_step > 0
        )

        if self._lisa_bistate_enabled:
            # Start in the alignment state, exactly as the authors' `init`.
            try:
                self._lisa.switch_state("alignment")
            except Exception:
                pass
            self._lisa.status = "alignment"

        # `clock` counts steps within the current state (resets on every
        # switch); `current_step` (in the wrapper) is the global counter.
        self._lisa_clock = 0

    # ------------------------------------------------------------------ #
    # Alignment-dataset plumbing
    # ------------------------------------------------------------------ #
    def _get_lisa_alignment_loader(self):
        """Lazily build the alignment dataloader (uses the trainer's collator)."""
        if self._lisa_alignment_loader is None:
            from torch.utils.data import DataLoader, RandomSampler

            dataset = self._lisa_alignment_dataset
            # Tolerate callers that pass an already-built DataLoader as the
            # ``alignment_dataset`` (the documented contract is a *dataset*,
            # but wrapping a DataLoader in another DataLoader makes its
            # ``dataset`` un-subscriptable and crashes the fetch). Use it as-is.
            if isinstance(dataset, DataLoader):
                self._lisa_alignment_loader = self.accelerator.prepare(dataset)
                return self._lisa_alignment_loader
            loader = DataLoader(
                dataset,
                batch_size=self._train_batch_size,
                sampler=RandomSampler(dataset),
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=self.args.dataloader_drop_last,
            )
            self._lisa_alignment_loader = self.accelerator.prepare(loader)
        return self._lisa_alignment_loader

    def _sample_from_alignment(self):
        """Draw the next alignment batch, cycling the dataloader when exhausted."""
        loader = self._get_lisa_alignment_loader()
        if self._lisa_alignment_iter is None:
            self._lisa_alignment_iter = iter(loader)
        try:
            return next(self._lisa_alignment_iter)
        except StopIteration:
            self._lisa_alignment_iter = iter(loader)
            return next(self._lisa_alignment_iter)

    def _check_mode(self, inputs):
        """Switch state on interval boundaries and pick the right batch.

        Mirrors the authors' ``check_mode``: when the per-state ``clock`` hits
        the state's interval, flip the state, commit the consensus checkpoint
        and reset the clock. While in the alignment state, the user batch is
        replaced with a freshly sampled alignment batch.
        """
        if not self._lisa_bistate_enabled:
            return inputs

        if self._lisa.status == "alignment":
            if (
                self._lisa_finetune_step > 0
                and self._lisa_clock % self._lisa_alignment_step == 0
                and self._lisa_clock != 0
            ):
                # alignment -> finetune
                self._lisa.switch_state("finetune")
                self._lisa_clock = 0
                # finetune state uses the user `train_dataset` batch (`inputs`).
            else:
                # stay in alignment: train on alignment data instead.
                inputs = self._sample_from_alignment()
        else:
            if (
                self._lisa_alignment_step > 0
                and self._lisa_clock % self._lisa_finetune_step == 0
                and self._lisa_clock != 0
            ):
                # finetune -> alignment
                self._lisa.switch_state("alignment")
                self._lisa_clock = 0
                # entering alignment: this step trains on alignment data.
                inputs = self._sample_from_alignment()
        return inputs

    # ------------------------------------------------------------------ #
    # Loss / step
    # ------------------------------------------------------------------ #
    def compute_loss(  # type: ignore[override]
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        try:
            result = super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        except TypeError:
            result = super().compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            base_loss, outputs = result
        else:
            base_loss = result
            outputs = None

        with self._lisa.apply_proximal_penalty(base_loss) as modified_loss:
            final_loss = modified_loss

        if return_outputs:
            return final_loss, outputs
        return final_loss

    def training_step(self, model, inputs, num_items_in_batch=None):  # type: ignore[override]
        # Bi-state: possibly switch state and swap the batch for an alignment
        # batch *before* the forward/backward pass (authors' check_mode).
        inputs = self._check_mode(inputs)

        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
        except TypeError:
            loss = super().training_step(model, inputs)

        # Advance global + per-state counters (authors' steps / clock).
        self._lisa.step()
        self._lisa_clock += 1

        return loss
