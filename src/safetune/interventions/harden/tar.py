"""
TAR: Tamper-Resistant Safeguards (Tamirisa et al., ICLR 2025,
arXiv:2408.00761). Repo: https://github.com/rishub-tamirisa/tamper-resistance

TAR is a meta-learning objective: it trains the model so that the
safety-behaviour loss remains low *even after* an adversary takes K SGD
steps on a harmful objective. The training loop has an outer SFT/retain
step and an inner simulated-adversary loop.

Original algorithm (paper Algorithm 1; repo ``modules/training.py`` ::
``tar_training_loop`` / ``inner_loop_step`` and ``modules/objectives.py``):

    for outer step:
        save_params = collect(model, mode="params")
        g_TR = 0
        for k = 1 .. K:                       # inner adversarial trajectory
            # adversary takes one harmful SGD step
            adversary_next_token_obj_step(model, harm_batch); inner_opt.step()
            # tamper-resistance loss at the *current tampered* parameters
            L_TR = tamper_resistance_obj(model, safety_batch)
            L_TR.backward()                   # NOT detached
            g_TR += (1/K) * collect(model, mode="grads")
            model.zero_grad()
        restore(model, save_params)           # back to original params
        retain_loss(model, retain_batch).backward()   # outer retain grad
        add g_TR back into model.grad         # meta-gradient
        optimizer.step()

The meta-gradient ``g_TR`` is a **first-order** estimator: the post-tamper
safety loss is evaluated at every inner step and its gradient (w.r.t. the
parameters *at that inner point*) is accumulated; the adversary update is
treated with a straight-through estimator (FO-MAML), so the TR gradient is
carried back to the original parameters one-to-one. The defining property
is that ``L_TR`` **contributes gradient** to the outer update — it must not
be ``.detach()``-ed.

We expose the *outer loss assembly* as a standalone function so callers can
plug their own optimizers and dataloaders. The returned scalar is
constructed so that ``outer_loss.backward()`` deposits exactly
``grad(retain_loss) + lambda_tar * g_TR`` onto ``model.parameters()``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TARConfig:
    """Configuration for the TAR outer loss.

    Attributes:
        inner_steps: K, the number of adversarial SGD steps the simulated
            tamperer takes per outer batch. ``5`` is the paper default for
            single-step robustness; ``25`` for the strong variant.
        inner_lr: inner-loop learning rate.
        lambda_tar: weight on the accumulated tamper-resistance meta-gradient
            in the outer objective. ``0.5`` to ``1.0`` is the paper's working
            range.
        accumulate_every_step: if ``True`` (paper default) the tamper-resistance
            loss is evaluated and its gradient accumulated at *every* one of the
            K inner steps and averaged by ``1/K``; if ``False`` it is evaluated
            only once at the final tampered parameters.
    """
    inner_steps: int = 5
    inner_lr: float = 1e-4
    lambda_tar: float = 1.0
    accumulate_every_step: bool = True


@torch.enable_grad()
def tar_outer_loss(
    model: nn.Module,
    retain_batch: Dict[str, torch.Tensor],
    harm_batch: Dict[str, torch.Tensor],
    safety_batch: Dict[str, torch.Tensor],
    task_loss_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
    config: Optional[TARConfig] = None,
    safety_loss_fn: Optional[
        Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor]
    ] = None,
) -> torch.Tensor:
    """Compute the TAR outer-loop loss for one batch triple.

    Implements the TAR first-order meta-learning objective (Tamirisa et al.,
    Algorithm 1). The simulated adversary takes ``K = config.inner_steps``
    harmful SGD steps; at each tampered point the tamper-resistance (safety)
    loss is evaluated and its gradient is accumulated into a meta-gradient
    ``g_TR``. The returned scalar is built so that ``backward()`` deposits

        grad(retain_loss) + lambda_tar * g_TR

    onto ``model.parameters()`` — i.e. the post-tamper safety loss **does**
    contribute gradient to the outer update (it is *not* detached, and the
    inner loop is *not* run under ``no_grad``).

    Args:
        model: the safeguarded model whose parameters the outer step updates.
        retain_batch: batch for the benign retain (capabilities) loss.
        harm_batch: batch the simulated adversary fine-tunes on.
        safety_batch: held-out batch for the tamper-resistance / safety loss.
        task_loss_fn: ``(model, batch) -> scalar``; used for the retain loss
            and the inner adversarial (harmful) loss.
        config: :class:`TARConfig`.
        safety_loss_fn: optional ``(model, batch) -> scalar`` for the
            tamper-resistance loss ``L_TR`` (paper uses negative-entropy or
            DPO loss here). Defaults to ``task_loss_fn`` so existing callers
            need not change anything.

    Returns:
        A differentiable scalar; after ``backward()`` the parameter grads are
        the TAR outer meta-gradient.
    """
    cfg = config or TARConfig()
    tr_loss_fn = safety_loss_fn or task_loss_fn

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("tar_outer_loss: model has no trainable parameters.")

    # Snapshot the original (outer) parameters; these are what the outer step
    # updates and what the meta-gradient must be expressed w.r.t.
    original = [p.detach().clone() for p in params]

    # ------------------------------------------------------------------
    # Inner adversarial trajectory + first-order tamper-resistance gradient.
    #
    # FO-MAML / straight-through: the adversary update is a detached SGD step,
    # but the tamper-resistance loss L_TR is differentiated *at each tampered
    # point* and its gradient is accumulated. Because consecutive tampered
    # states differ from the originals only by a detached offset, the gradient
    # of L_TR w.r.t. the tampered params equals (first order) the gradient
    # w.r.t. the originals — so we accumulate it directly into g_TR.
    # ------------------------------------------------------------------
    g_tr = [torch.zeros_like(p) for p in params]
    n_inner = max(int(cfg.inner_steps), 0)
    
    # how many times L_TR is measured along the trajectory
    n_measure = n_inner if (cfg.accumulate_every_step and n_inner > 0) else 1
    measure_weight = 1.0 / float(n_measure)

    post_tamper_value = torch.zeros(())  # diagnostic only

    for k in range(max(n_inner, 1)):
        if n_inner > 0:
            # --- adversary harmful SGD step (detached: straight-through) ---
            model.zero_grad(set_to_none=True)
            h_loss = task_loss_fn(model, harm_batch)
            h_grads = torch.autograd.grad(
                h_loss, params, retain_graph=False, allow_unused=True
            )
            with torch.no_grad():
                for p, g in zip(params, h_grads):
                    if g is not None:
                        p.add_(g, alpha=-cfg.inner_lr)

        # --- tamper-resistance loss at the current (tampered) parameters ---
        # Evaluate at every inner step (paper default) or only at the end.
        measure_now = cfg.accumulate_every_step or (k == max(n_inner, 1) - 1)
        if measure_now:
            model.zero_grad(set_to_none=True)
            tr_loss = tr_loss_fn(model, safety_batch)  # NOT detached
            tr_grads = torch.autograd.grad(
                tr_loss, params, retain_graph=False, allow_unused=True
            )
            with torch.no_grad():
                for acc, g in zip(g_tr, tr_grads):
                    if g is not None:
                        acc.add_(g, alpha=measure_weight)
                post_tamper_value = tr_loss.detach()

    # ------------------------------------------------------------------
    # Restore the original parameters; the outer step is taken from there.
    # ------------------------------------------------------------------
    with torch.no_grad():
        for p, orig in zip(params, original):
            p.copy_(orig)
    model.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # Outer loss assembly.
    #
    # retain_loss is differentiable w.r.t. the restored params. We splice the
    # accumulated tamper-resistance meta-gradient ``g_TR`` into the autograd
    # graph via a surrogate inner-product term: d/dp [ sum <p, g_TR> ] = g_TR.
    # ``g_TR`` is a constant tensor here (detached), so backward() deposits
    #     grad(retain_loss) + lambda_tar * g_TR
    # exactly the TAR outer meta-gradient.
    # ------------------------------------------------------------------
    retain_loss = task_loss_fn(model, retain_batch)

    tr_surrogate = sum(
        (p * g.detach()).sum() for p, g in zip(params, g_tr)
    )

    outer_loss = retain_loss + cfg.lambda_tar * tr_surrogate

    # Attach diagnostics (do not affect autograd).
    outer_loss.tar_retain_loss = retain_loss.detach()  # type: ignore[attr-defined]
    outer_loss.tar_post_tamper_safety = post_tamper_value  # type: ignore[attr-defined]

    return outer_loss


__all__ = ["TARConfig", "tar_outer_loss"]