"""
NPO: Negative Preference Optimization (Zhang et al., "Negative Preference
Optimization: From Catastrophic Collapse to Effective Unlearning", 2024,
arXiv:2404.05868).
Authors' reference implementation:
https://github.com/licong-lin/negative-preference-optimization

Algorithm (faithful to the authors' ``CustomTrainerForgetting.compute_loss``
in ``TOFU/dataloader.py``):

  NPO is a DPO-style loss that keeps only the *negative* (forget) term.
  Against a frozen reference (oracle) model it drives down the likelihood of
  the forget set without the catastrophic collapse of plain gradient ascent.

  Per the authors, with the per-sequence negative log-likelihood
  ``L(y) = -log p(y)`` (summed cross-entropy over the answer tokens)::

      neg_log_ratios = L_theta(y) - L_ref(y)
                     = -log p_theta(y) + log p_ref(y)
                     = -(log p_theta(y) - log p_ref(y))

      L_NPO = -(2/beta) * E[ logsigmoid(beta * neg_log_ratios) ]
            =  (2/beta) * E[ -logsigmoid(-beta*(log p_theta(y) - log p_ref(y))) ]

  which is exactly the paper's
  ``L_NPO = (2/beta)*E[ logsigmoid(-beta*(logp_theta - logp_ref)) ]`` up to the
  sign convention of a minimised loss. The authors' literal line is::

      loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

  As ``beta -> 0`` NPO recovers gradient ascent; for ``beta > 0`` the
  logsigmoid saturates once the forget likelihood is already low, which is
  what tames the collapse.

  Retain-term variants (authors' ``loss_type`` branches):

  * ``"npo"``            -- bare NPO forget loss, no retain term.
  * ``"npo_grad_diff"``  -- NPO + standard cross-entropy on the retain set
                            (gradient-difference style)::
        loss = npo_coeff*L_NPO + grad_diff_coeff*CE_retain
  * ``"npo_KL"``         -- NPO + KL(student || frozen-reference) on the
                            retain set::
        loss = npo_coeff*L_NPO + KL_coeff*KL_retain
    with ``KL_retain = kl_div(log_softmax(student), log_softmax(ref),
    reduction="batchmean", log_target=True)``.

  The authors' TOFU defaults are ``beta=0.1``, ``npo_coeff=1.0``,
  ``grad_diff_coeff=1.0``, ``KL_coeff=1.0``, ``lr=1e-5``, ``num_epochs=10``,
  ``weight_decay=0.01``, with a fine-tuned reference policy.

For safety unlearning we set:
  * F = harmful examples whose response patterns we want to forget.
  * R = benign examples we want to keep performance on.

The public entry point ``npo_unlearn`` takes a model, a frozen reference (or
None to snapshot the model itself), and iterables over forget / retain
batches, and runs the NPO schedule in place.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetune._refusal_helpers import _get_decoder_layers  # noqa: F401 (re-exported only)

logger = logging.getLogger(__name__)

# Loss-type variants, matching the authors' ``loss_type`` strings.
_VARIANTS = ("npo", "npo_grad_diff", "npo_KL")


@dataclass
class NPOConfig:
    """Configuration for NPO unlearning.

    Attributes:
        variant: which retain-term variant to run -- ``"npo"`` (forget term
            only), ``"npo_grad_diff"`` (NPO + retain cross-entropy), or
            ``"npo_KL"`` (NPO + retain KL toward the frozen reference).
        beta: NPO temperature. ``0.1`` is the authors' TOFU default. As
            ``beta -> 0`` NPO recovers gradient ascent; larger ``beta``
            saturates the logsigmoid sooner.
        npo_coeff: weight on the NPO forget term. ``1.0`` is the authors'
            default. Only used by the ``npo_grad_diff`` / ``npo_KL`` variants
            (the bare ``npo`` variant always uses the un-weighted loss).
        grad_diff_coeff: weight on the retain cross-entropy term in the
            ``npo_grad_diff`` variant. ``1.0`` is the authors' default.
        kl_coeff: weight on the retain KL term in the ``npo_KL`` variant.
            ``1.0`` is the authors' default.
        num_epochs: number of full passes over the forget iterable (and, for
            the retain variants, the retain iterable). ``10`` is the authors'
            TOFU default.
        lr: optimizer learning rate. ``1e-5`` is the authors' default.
        weight_decay: optimizer weight decay. ``0.01`` is the authors'
            default.
        optimizer: ``"adamw"`` (authors' default via HF ``Trainer``) or
            ``"sgd"``.
        max_steps: optional hard cap on the total number of optimizer steps
            across all epochs. ``None`` runs the full epoch schedule.
    """

    variant: str = "npo"
    beta: float = 0.1
    npo_coeff: float = 1.0
    grad_diff_coeff: float = 1.0
    kl_coeff: float = 1.0
    num_epochs: int = 10
    lr: float = 1e-5
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    max_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.variant not in _VARIANTS:
            raise ValueError(
                f"NPOConfig.variant must be one of {_VARIANTS}, got {self.variant!r}"
            )


def get_batch_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sequence summed cross-entropy, the authors' ``get_batch_loss``.

    Faithful to ``TOFU/data_module.py``::

        shifted_labels = labels[..., 1:].contiguous()
        output = output[..., :-1, :].contiguous()
        loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    Returns a ``(batch,)`` tensor: for each sequence, the *sum* of the token
    cross-entropies over its answer tokens (i.e. the negative log-likelihood
    ``-log p(y)`` of that sequence).
    """
    shifted_labels = labels[..., 1:].contiguous()
    shifted_logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    # transpose(-1,-2): CrossEntropyLoss wants (batch, classes, seq).
    per_token = loss_function(shifted_logits.transpose(-1, -2), shifted_labels)
    return per_token.sum(dim=-1)


def npo_forget_loss(
    forget_loss_current: torch.Tensor,
    forget_loss_ref: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """The bare NPO forget loss, the authors' literal expression.

    ``forget_loss_current`` / ``forget_loss_ref`` are per-sequence negative
    log-likelihoods (``get_batch_loss`` outputs) from the trained model and
    the frozen reference respectively.

    Faithful to ``CustomTrainerForgetting.compute_loss``::

        neg_log_ratios = forget_loss_current - forget_loss_oracle
        loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

    Note ``neg_log_ratios = L_theta - L_ref = -(logp_theta - logp_ref)``, so
    this equals the paper's
    ``L_NPO = (2/beta)*E[ logsigmoid(-beta*(logp_theta - logp_ref)) ]``.
    """
    neg_log_ratios = forget_loss_current - forget_loss_ref
    return -F.logsigmoid(beta * neg_log_ratios).mean() * 2.0 / beta


def _retain_kl(
    student_logits: torch.Tensor, ref_logits: torch.Tensor
) -> torch.Tensor:
    """KL(student || reference) on the retain set, the authors' ``npo_KL`` term.

    Faithful to ``CustomTrainerForgetting.compute_loss``::

        retain_probs  = F.log_softmax(retain_outputs.logits, dim=-1).view(-1, V)
        current_probs = F.log_softmax(current_outputs.logits, dim=-1).view(-1, V)
        retain_loss = nn.functional.kl_div(current_probs, retain_probs,
                                           reduction='batchmean', log_target=True)
    """
    V = student_logits.size(-1)
    current_probs = F.log_softmax(student_logits, dim=-1).reshape(-1, V)
    ref_probs = F.log_softmax(ref_logits, dim=-1).reshape(-1, V)
    return F.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    )


def _build_optimizer(model: nn.Module, cfg: NPOConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def npo_unlearn(
    model: nn.Module,
    forget_batches: Iterable[Dict[str, torch.Tensor]],
    retain_batches: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
    *,
    reference: Optional[nn.Module] = None,
    forward_fn: Optional[
        Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor]
    ] = None,
    config: Optional[NPOConfig] = None,
) -> nn.Module:
    """Run NPO unlearning in place on ``model``.

    Faithful to the authors' ``CustomTrainerForgetting``: each step computes
    the NPO forget loss (:func:`npo_forget_loss`) against the frozen
    ``reference`` model, optionally adds a retain term, and takes one
    optimizer step.

    The training schedule pairs the forget and retain iterables step-for-step
    (``zip``) for ``cfg.num_epochs`` epochs, mirroring the authors' batched
    ``(forget_inputs, retain_inputs)`` tuples.

    Args:
        model: the model to unlearn. Updated in place.
        forget_batches: iterable yielding batches of forget examples. Each
            batch must carry ``labels`` (answer tokens, ``-100`` for masked
            positions). Re-iterated once per epoch, so pass a re-iterable
            (list / ``DataLoader``) for ``num_epochs > 1``.
        retain_batches: iterable yielding batches of retain examples; required
            for the ``npo_grad_diff`` and ``npo_KL`` variants, ignored for the
            bare ``npo`` variant. Also re-iterated once per epoch.
        reference: the frozen reference (oracle) model. If ``None``, a
            deepcopy of ``model`` is snapshotted once at start (the authors'
            ``fine_tuned`` reference policy).
        forward_fn: callable ``(model, batch) -> logits``. If ``None``,
            ``model(**batch).logits`` is used.
        config: :class:`NPOConfig`.

    Returns:
        The updated model (same object, mutated in place).
    """
    cfg = config or NPOConfig()

    if cfg.variant != "npo" and retain_batches is None:
        raise ValueError(
            f"variant {cfg.variant!r} requires retain_batches; got None"
        )

    if reference is None:
        reference = copy.deepcopy(model)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad = False

    def _logits(m: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if forward_fn is not None:
            return forward_fn(m, batch)
        # Strip labels: passing labels causes HF to compute the loss
        # internally and return it alongside logits, corrupting our own
        # per-sequence get_batch_loss computation. We compute the loss
        # ourselves from the raw logits.
        fwd_batch = {k: v for k, v in batch.items() if k != "labels"}
        out = m(**fwd_batch)
        return out.logits if hasattr(out, "logits") else out

    opt = _build_optimizer(model, cfg)

    total_steps = 0
    max_steps = cfg.max_steps  # None => unbounded

    def _budget_left() -> bool:
        return max_steps is None or total_steps < max_steps

    epochs = max(1, cfg.num_epochs)
    for epoch in range(1, epochs + 1):
        if not _budget_left():
            break
        steps = 0

        if cfg.variant == "npo":
            iterator: Iterable = ((f, None) for f in forget_batches)
        else:
            iterator = zip(forget_batches, retain_batches)  # type: ignore[arg-type]

        for f_batch, r_batch in iterator:
            if not _budget_left():
                break
            opt.zero_grad(set_to_none=True)

            # --- NPO forget term -------------------------------------------
            f_logits = _logits(model, f_batch)
            forget_loss_current = get_batch_loss(f_logits, f_batch["labels"])
            with torch.no_grad():
                f_logits_ref = _logits(reference, f_batch)
                forget_loss_ref = get_batch_loss(
                    f_logits_ref, f_batch["labels"]
                )
            forget_loss = npo_forget_loss(
                forget_loss_current, forget_loss_ref, cfg.beta
            )

            # --- retain term + combination ---------------------------------
            if cfg.variant == "npo":
                # Authors' bare branch: loss = npo_forget_loss (un-weighted).
                loss = forget_loss
            elif cfg.variant == "npo_grad_diff":
                r_logits = _logits(model, r_batch)  # type: ignore[arg-type]
                retain_loss = F.cross_entropy(
                    r_logits[..., :-1, :].reshape(-1, r_logits.size(-1)),
                    r_batch["labels"][..., 1:].reshape(-1),  # type: ignore[index]
                    ignore_index=-100,
                )
                loss = (
                    cfg.npo_coeff * forget_loss
                    + cfg.grad_diff_coeff * retain_loss
                )
            else:  # cfg.variant == "npo_KL"
                r_logits = _logits(model, r_batch)  # type: ignore[arg-type]
                with torch.no_grad():
                    r_logits_ref = _logits(reference, r_batch)  # type: ignore[arg-type]
                retain_loss = _retain_kl(r_logits, r_logits_ref)
                loss = (
                    cfg.npo_coeff * forget_loss + cfg.kl_coeff * retain_loss
                )

            loss.backward()
            opt.step()
            steps += 1
            total_steps += 1

        logger.info(
            "npo_unlearn: epoch %d/%d (%s) -- %d step(s).",
            epoch,
            epochs,
            cfg.variant,
            steps,
        )

    logger.info(
        "npo_unlearn: completed (%d total optimizer steps, variant=%s).",
        total_steps,
        cfg.variant,
    )
    return model


__all__ = [
    "NPOConfig",
    "npo_unlearn",
    "npo_forget_loss",
    "get_batch_loss",
]