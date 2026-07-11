"""
Gradient-Ascent (GA) unlearning and its Gradient-Difference (GradDiff)
variant -- the canonical TOFU unlearning baselines.

Reference: Maini et al., "TOFU: A Task of Fictitious Unlearning for LLMs"
(arXiv:2401.06121). Authors' reference implementation:
https://github.com/locuslab/tofu -- ``dataloader.py``, the
``CustomTrainerForgetting.compute_loss`` method, ``forget_loss`` types
``grad_ascent`` / ``grad_diff`` / ``KL``.

Algorithm (faithful to the authors' ``compute_loss`` branches):

  Let ``CE(D)`` be the standard next-token cross-entropy loss the model
  reports as ``outputs.loss`` on a batch of dataset ``D`` (mean over
  non-masked label tokens, ``ignore_index = -100``).

  * ``grad_ascent`` -- on the forget set F only::

        forget_loss = outputs.loss          # CE(F)
        forget_loss = forget_loss * -1      # negate
        loss = forget_loss                  # = -CE(F)

    Minimising ``-CE(F)`` is gradient *ascent* on the forget data: it
    pushes the model AWAY from the forget targets.

  * ``grad_diff`` -- forget set F plus retain set R::

        forget_loss = outputs.loss * -1     # -CE(F)
        retain_loss = retain_outputs.loss   #  CE(R)   (same model)
        loss = forget_loss + retain_loss    # -CE(F) + CE(R)

    The retain term ``CE(R)`` is an ordinary fine-tuning loss that anchors
    the model on data we want to keep, counteracting the collapse that
    pure ascent causes.

  * ``KL`` -- forget set F plus a KL retain anchor against a frozen
    reference (the authors' ``oracle_model``)::

        forget_loss = outputs.loss * -1     # -CE(F)
        # current model and oracle, both on the retain batch:
        current_probs = log_softmax(current_outputs.logits, dim=-1)
        retain_probs  = log_softmax(oracle_outputs.logits,  dim=-1)
        retain_loss = kl_div(current_probs, retain_probs,
                             reduction='batchmean', log_target=True)
        loss = forget_loss + retain_loss    # -CE(F) + KL(current||oracle | R)

    Here the retain anchor keeps the *distribution* on R close to the
    original model rather than re-fitting hard labels.

The authors apply no extra collapse heuristic beyond the retain term that
``grad_diff`` / ``KL`` add; ``grad_ascent`` runs bare negative CE. We match
that exactly -- the only optional, off-by-default safety knob is
``forget_clip`` (mirroring SCRUB's identically-named opt-in), which caps the
*magnitude* of the per-batch forget CE before negation.

For safety unlearning we set:
  * F = harmful examples whose response patterns we want to forget.
  * R = benign examples we want to keep performance on.

The public entry point :func:`gradient_ascent_unlearn` takes a model, an
iterable over forget batches, an optional iterable over retain batches, and
an optional frozen reference (for the ``KL`` variant), and runs the chosen
forget-loss schedule in place.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The three faithful TOFU forget-loss types implemented here.
_FORGET_LOSS_TYPES = ("grad_ascent", "grad_diff", "KL")


@dataclass
class GradientAscentConfig:
    """Configuration for Gradient-Ascent / Gradient-Difference unlearning.

    Attributes:
        forget_loss: which TOFU forget-loss variant to run. One of
            ``"grad_ascent"`` (``loss = -CE(F)``), ``"grad_diff"``
            (``loss = -CE(F) + CE(R)``) or ``"KL"``
            (``loss = -CE(F) + KL(model||reference | R)``).
        epochs: number of passes over the forget iterable. The retain
            iterable (when used) is consumed in lock-step, one retain batch
            per forget batch -- exactly the authors' paired
            ``(forget_inputs, retain_inputs)`` collator. Re-iterated once
            per epoch, so pass re-iterables (lists / ``DataLoader``s).
        lr: optimizer learning rate. ``1e-5`` matches the TOFU finetune /
            forget learning rate.
        weight_decay: optimizer weight decay (TOFU default ``0.01``).
        optimizer: ``"adamw"`` (the authors' default) or ``"sgd"``.
        forget_clip: optional cap on the *magnitude* of the per-batch
            forget cross-entropy before it is negated. ``None`` (the TOFU
            default) disables the clip -- pure ascent. An opt-in safety
            knob only; the authors invent no such heuristic.
        max_steps: optional hard cap on the total number of optimizer
            steps across all epochs. ``None`` runs the full schedule.
    """

    forget_loss: str = "grad_ascent"
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    forget_clip: Optional[float] = None
    max_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.forget_loss not in _FORGET_LOSS_TYPES:
            raise ValueError(
                f"forget_loss must be one of {_FORGET_LOSS_TYPES}, "
                f"got {self.forget_loss!r}"
            )


def _ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy, the value HF reports as ``outputs.loss``.

    Causal-LM shift: predict token ``t+1`` from position ``t``. Mean over
    non-masked label tokens with ``ignore_index = -100``.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def _build_optimizer(
    model: nn.Module, cfg: GradientAscentConfig
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def gradient_ascent_unlearn(
    model: nn.Module,
    forget_batches: Iterable[Dict[str, torch.Tensor]],
    retain_batches: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
    *,
    reference: Optional[nn.Module] = None,
    forward_fn: Optional[
        Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor]
    ] = None,
    config: Optional[GradientAscentConfig] = None,
) -> nn.Module:
    """Run Gradient-Ascent / Gradient-Difference unlearning in place.

    Faithful to TOFU's ``CustomTrainerForgetting.compute_loss``:

    * ``grad_ascent``: ``loss = -CE(forget)``.
    * ``grad_diff``: ``loss = -CE(forget) + CE(retain)`` -- the retain CE
      is computed with the *same* (currently-training) model.
    * ``KL``: ``loss = -CE(forget) + KL(model || reference | retain)``,
      where ``reference`` is the frozen oracle and the KL is
      ``kl_div(log_softmax(model), log_softmax(reference),
      reduction='batchmean', log_target=True)``.

    Args:
        model: the model to unlearn. Updated in place.
        forget_batches: iterable of forget-set batches. Re-iterated once
            per epoch -- pass a re-iterable for ``epochs > 1``.
        retain_batches: iterable of retain-set batches, required for
            ``grad_diff`` and ``KL`` and ignored for ``grad_ascent``.
            Consumed one batch per forget batch (the authors' paired
            collator). Re-iterated once per epoch.
        reference: frozen oracle model, required for the ``KL`` variant.
            If ``None`` for ``KL``, a deepcopy of ``model`` is snapshotted
            once at start. Unused by the other variants.
        forward_fn: callable ``(model, batch) -> logits``. If ``None``,
            ``model(**batch).logits`` is used and ``batch['labels']``
            drives the cross-entropy.
        config: :class:`GradientAscentConfig`.

    Returns:
        The updated model (same object, mutated in place).
    """
    cfg = config or GradientAscentConfig()
    needs_retain = cfg.forget_loss in ("grad_diff", "KL")
    if needs_retain and retain_batches is None:
        raise ValueError(
            f"forget_loss={cfg.forget_loss!r} requires retain_batches."
        )

    if cfg.forget_loss == "KL":
        if reference is None:
            reference = copy.deepcopy(model)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad = False

    def _logits(m: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if forward_fn is not None:
            return forward_fn(m, batch)
        out = m(**{k: v for k, v in batch.items() if k != "labels"})
        return out.logits if hasattr(out, "logits") else out

    def _forget_term(f_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Authors' ``forget_loss = outputs.loss * -1``."""
        f_logits = _logits(model, f_batch)
        forget_ce = _ce_loss(f_logits, f_batch["labels"])
        if cfg.forget_clip is not None:
            forget_ce = forget_ce.clamp(max=cfg.forget_clip)
        return forget_ce * -1.0

    def _retain_ce_term(r_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Authors' grad_diff ``retain_loss = retain_outputs.loss``."""
        r_logits = _logits(model, r_batch)
        return _ce_loss(r_logits, r_batch["labels"])

    def _retain_kl_term(r_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Authors' KL ``retain_loss``: KL(current || oracle) on retain."""
        current_logits = _logits(model, r_batch)
        with torch.no_grad():
            oracle_logits = _logits(reference, r_batch)
        current_probs = F.log_softmax(current_logits, dim=-1)
        retain_probs = F.log_softmax(oracle_logits, dim=-1)
        # kl_div(input=current, target=oracle, log_target=True): the exact
        # argument order of the authors' nn.functional.kl_div call.
        return F.kl_div(
            current_probs,
            retain_probs,
            reduction="batchmean",
            log_target=True,
        )

    opt = _build_optimizer(model, cfg)
    total_steps = 0
    max_steps = cfg.max_steps

    epochs = max(1, cfg.epochs)
    for epoch in range(1, epochs + 1):
        if max_steps is not None and total_steps >= max_steps:
            break
        steps = 0
        retain_iter = iter(retain_batches) if needs_retain else None
        for f_batch in forget_batches:
            if max_steps is not None and total_steps >= max_steps:
                break
            opt.zero_grad(set_to_none=True)
            loss = _forget_term(f_batch)

            if needs_retain:
                try:
                    r_batch = next(retain_iter)  # type: ignore[arg-type]
                except StopIteration:
                    # Retain iterable exhausted before forget: the authors'
                    # paired collator always supplies one retain batch per
                    # forget batch. Stop the epoch rather than fabricate.
                    logger.warning(
                        "gradient_ascent_unlearn: retain iterable exhausted "
                        "mid-epoch %d; ending epoch early.",
                        epoch,
                    )
                    break
                if cfg.forget_loss == "grad_diff":
                    loss = loss + _retain_ce_term(r_batch)
                else:  # KL
                    loss = loss + _retain_kl_term(r_batch)

            loss.backward()
            opt.step()
            steps += 1
            total_steps += 1

        logger.info(
            "gradient_ascent_unlearn: epoch %d/%d -- %d step(s), loss_type=%s.",
            epoch,
            epochs,
            steps,
            cfg.forget_loss,
        )

    logger.info(
        "gradient_ascent_unlearn: completed (%d total optimizer steps).",
        total_steps,
    )
    return model


__all__ = ["GradientAscentConfig", "gradient_ascent_unlearn"]
