"""
SCRUB: SCalable Remembering and Unlearning unBound (Kurmanji et al.,
"Towards Unbounded Machine Unlearning", NeurIPS 2023, arXiv:2302.09880).
Authors' reference implementation: https://github.com/meghdadk/SCRUB

Algorithm (faithful to the authors' ``train_distill`` loop):

  Teacher-student distillation. The student starts as a copy of the trained
  model; a frozen teacher is the same model. SCRUB alternates two kinds of
  passes, controlled by an epoch schedule:

  * MAX-STEP (forget pass) -- on the forget set F, *maximize* the KL
    divergence between the student and the teacher::

        loss = -KL(student || teacher | F)

    This pushes the student's distribution AWAY from the teacher on the
    examples to be forgotten. No task / cross-entropy term is used here --
    only the negative divergence (authors' ``split == "maximize"`` branch).

  * MIN-STEP (retain pass) -- on the retain set R, *minimize* a weighted sum
    of the task cross-entropy and the KL toward the teacher::

        loss = gamma * CE(student | R) + alpha * KL(student || teacher | R)

    This pulls the student back toward the teacher on the retain set and
    keeps its task performance (authors' ``split == "minimize"`` branch;
    ``loss = gamma*loss_cls + alpha*loss_div + beta*loss_kd`` with
    ``loss_kd == 0`` for the plain-KD distiller SCRUB uses).

  The authors alternate these in an epoch schedule: for the first
  ``msteps`` epochs each epoch runs a *full* max-step pass over F followed
  by a *full* min-step pass over R; after ``msteps`` epochs only the
  min-step pass runs. The learning rate decays on a milestone schedule.

For safety unlearning we set:
  * R = benign examples we want to keep performance on.
  * F = harmful examples whose response patterns we want to forget.

The public entry point ``scrub_unlearn`` takes a model, a teacher (or None
to snapshot the model itself), and iterables over retain / forget batches,
and runs the SCRUB epoch schedule in place.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetune._refusal_helpers import _get_decoder_layers  # noqa: F401 (re-exported only)

logger = logging.getLogger(__name__)


@dataclass
class SCRUBConfig:
    """Configuration for SCRUB unlearning.

    Attributes:
        alpha: weight on the retain KL term (``loss_div`` on R). ``1.0`` is
            the authors' default.
        gamma: weight on the retain task cross-entropy term (``loss_cls``
            on R). ``1.0`` is the authors' default. Set to ``0`` if the
            retain batches carry no ``labels``.
        beta: weight on the forget divergence term (the ``-KL`` max-step).
            ``1.0`` reproduces the authors' bare ``loss = -loss_div``;
            values != 1 simply rescale the forget gradient.
        temperature: softmax temperature for the distillation KL.
            ``4.0`` is the authors' default (``opt.kd_T``).
        sgda_epochs: total number of SCRUB epochs. Each epoch is one full
            pass over the retain iterable (and, while ``epoch <= msteps``,
            one full pass over the forget iterable first).
        msteps: number of leading epochs that include the forget max-step.
            After ``msteps`` epochs only the retain min-step runs.
        lr: SGDA learning rate.
        lr_decay_epochs: epochs (1-indexed) at which the lr is multiplied by
            ``lr_decay_rate``.
        lr_decay_rate: multiplicative lr decay applied at each milestone.
        weight_decay: optimizer weight decay (``sgda_weight_decay``).
        momentum: SGD momentum (``sgda_momentum``).
        optimizer: ``"sgd"`` (authors' default) or ``"adamw"``.
        forget_clip: optional cap on the *magnitude* of the forget-step
            divergence to guard against runaway updates. ``None`` (the
            paper default) disables the clip. Kept as an opt-in safety knob.
        max_steps: optional hard cap on the total number of optimizer steps
            across all epochs/passes. ``None`` runs the full epoch schedule.
            Retained for backwards compatibility with callers that passed a
            step budget.
    """

    alpha: float = 1.0
    gamma: float = 1.0
    beta: float = 1.0
    temperature: float = 4.0
    sgda_epochs: int = 5
    msteps: int = 2
    lr: float = 5e-4
    lr_decay_epochs: List[int] = field(default_factory=lambda: [3, 4])
    lr_decay_rate: float = 0.1
    weight_decay: float = 0.1
    momentum: float = 0.9
    optimizer: str = "sgd"
    forget_clip: Optional[float] = None
    max_steps: Optional[int] = None


def _kl_div(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    """Temperature-scaled KL(student || teacher), the authors' ``DistillKL``.

    ``loss = KL(p_t || p_s) * T^2`` with ``p_s = log_softmax(s/T)`` and
    ``p_t = softmax(t/T)`` -- identical to RepDistiller's ``DistillKL``.
    """
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def _build_optimizer(model: nn.Module, cfg: SCRUBConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(
        params,
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )


def scrub_unlearn(
    model: nn.Module,
    retain_batches: Iterable[Dict[str, torch.Tensor]],
    forget_batches: Iterable[Dict[str, torch.Tensor]],
    *,
    teacher: Optional[nn.Module] = None,
    forward_fn: Optional[Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor]] = None,
    config: Optional[SCRUBConfig] = None,
) -> nn.Module:
    """Run SCRUB unlearning in place on ``model``..."""
    
    cfg = config or SCRUBConfig()
    
    # 1. Identify where the model lives
    target_device = next(model.parameters()).device

    if teacher is None:
        teacher = copy.deepcopy(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    def _sanitize_batch(batch: Dict[str, any]) -> Dict[str, torch.Tensor]:
        clean_batch = {}
        for k, v in batch.items():
            tensor_v = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            if tensor_v.dim() == 1:
                tensor_v = tensor_v.unsqueeze(0)
            clean_batch[k] = tensor_v.to(target_device)
        return clean_batch

    def _logits(m: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if forward_fn is not None:
            return forward_fn(m, batch)
        out = m(**{k: v for k, v in batch.items() if k != "labels"})
        return out.logits if hasattr(out, "logits") else out

    opt = _build_optimizer(model, cfg)
    decay_epochs = set(cfg.lr_decay_epochs or [])

    total_steps = 0
    max_steps = cfg.max_steps  # None => unbounded

    def _budget_left() -> bool:
        return max_steps is None or total_steps < max_steps

    def _forget_pass(epoch: int) -> int:
        """One full max-step pass over the forget set. Returns #steps taken."""
        nonlocal total_steps
        steps = 0
        for f_batch in forget_batches:
            if not _budget_left():
                break
                
            # 3. Sanitize the batch before passing it to the models
            f_batch = _sanitize_batch(f_batch)
            
            opt.zero_grad(set_to_none=True)
            s_f = _logits(model, f_batch)
            with torch.no_grad():
                t_f = _logits(teacher, f_batch)
            forget_div = _kl_div(s_f, t_f, cfg.temperature)
            if cfg.forget_clip is not None:
                forget_div = forget_div.clamp(max=cfg.forget_clip)
            # Authors' "maximize" branch: loss = -loss_div.
            loss = -cfg.beta * forget_div
            loss.backward()
            opt.step()
            steps += 1
            total_steps += 1
        return steps

    def _retain_pass(epoch: int) -> int:
        """One full min-step pass over the retain set. Returns #steps taken."""
        nonlocal total_steps
        steps = 0
        for r_batch in retain_batches:
            if not _budget_left():
                break
                
            # 4. Sanitize the retain batch too
            r_batch = _sanitize_batch(r_batch)
            
            opt.zero_grad(set_to_none=True)
            s_r = _logits(model, r_batch)
            
            with torch.no_grad():
                t_r = _logits(teacher, r_batch)
            retain_kl = _kl_div(s_r, t_r, cfg.temperature)

            # Authors' "minimize" branch: gamma*loss_cls + alpha*loss_div.
            loss = cfg.alpha * retain_kl
            if cfg.gamma != 0.0 and "labels" in r_batch:
                retain_ce = F.cross_entropy(
                    s_r.reshape(-1, s_r.size(-1)),
                    r_batch["labels"].reshape(-1),
                    ignore_index=-100,
                )
                loss = loss + cfg.gamma * retain_ce

            loss.backward()
            opt.step()
            steps += 1
            total_steps += 1
        return steps

    epochs = max(1, cfg.sgda_epochs)
    for epoch in range(1, epochs + 1):
        if not _budget_left():
            break
        # Max-steps (forget) only for the first `msteps` epochs.
        if epoch <= cfg.msteps:
            f_steps = _forget_pass(epoch)
        else:
            f_steps = 0
        # Min-step (retain) pass always runs.
        r_steps = _retain_pass(epoch)
        logger.info(
            "scrub_unlearn: epoch %d/%d -- %d forget step(s), %d retain step(s).",
            epoch,
            epochs,
            f_steps,
            r_steps,
        )
        # Learning-rate decay milestones.
        if epoch in decay_epochs:
            for g in opt.param_groups:
                g["lr"] *= cfg.lr_decay_rate

    logger.info("scrub_unlearn: completed (%d total optimizer steps).", total_steps)
    return model


__all__ = ["SCRUBConfig", "scrub_unlearn"]
