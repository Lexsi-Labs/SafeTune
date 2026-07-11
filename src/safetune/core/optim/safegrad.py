"""
SafeGrad: Gradient Surgery for Safe LLM Fine-Tuning.

Reference: https://arxiv.org/abs/2508.07172 (SafeGrad paper, Aug 2025).

The paper diagnoses that fine-tuning failures stem from *conflicting gradients*:
the user-task update vector and the safety-alignment update vector can have
negative cosine, in which case stepping along the user-task gradient
undermines safety. SafeGrad's fix is two-fold:

  1. Compute the alignment gradient ``g_align``. Two modes:
     a) ``safety_dataset`` mode: backpropagate a cross-entropy / SFT loss on
        a small alignment dataset. This is what
        :class:`SafeGradProjector.capture_safety_gradients` consumes.
     b) ``kl_alignment`` mode (paper's headline contribution):
        backpropagate ``KL(student_logits || frozen_aligned_logits)`` on
        the user batch, capturing the distributional safety profile of the
        aligned foundation model. See :func:`safegrad_kl_alignment_loss`.

  2. At training time, project the user-task gradient ``g_task`` away from
     the conflicting component:

        if g_task . g_align < 0:
            g_task_safe = g_task - (g_task . g_align / |g_align|^2) * g_align
        else:
            g_task_safe = g_task   # no conflict, leave it alone

This module provides the gradient-surgery primitive
(:class:`SafeGradProjector`) and the KL-alignment loss helper
(:func:`safegrad_kl_alignment_loss`). The HF Trainer subclass at
``safetune.harden.safegrad.SafeGradTrainer`` composes them.
"""

import logging
from typing import Dict

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def safegrad_kl_alignment_loss(
    student_logits: torch.Tensor,
    aligned_logits: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """KL(student || aligned) safety distillation loss (paper's KL mode).

    Use this when you have access to a *frozen aligned* model alongside the
    model being fine-tuned. Run both on the user batch, then call this on
    their per-token logits to capture the aligned model's distributional
    refusal profile. The resulting scalar is the alignment objective whose
    gradient is fed to :meth:`SafeGradProjector.capture_safety_gradients`.

    Args:
        student_logits: logits from the model being fine-tuned, shape
            ``(batch, seq, vocab)``.
        aligned_logits: logits from the frozen aligned reference, same shape.
            Should be detached so no grad flows back to the reference.
        temperature: softmax temperature. ``1.0`` is the paper default.
        reduction: passed to ``F.kl_div``.

    Returns:
        Scalar KL loss. ``loss.backward()`` populates ``student.grad`` with
        the alignment-direction gradient.
    """
    if student_logits.shape != aligned_logits.shape:
        raise ValueError(
            f"safegrad_kl_alignment_loss: shape mismatch student={tuple(student_logits.shape)} "
            f"vs aligned={tuple(aligned_logits.shape)}"
        )
    log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
    p_aligned = F.softmax(aligned_logits.detach() / temperature, dim=-1)
    return F.kl_div(log_p_student, p_aligned, reduction=reduction) * (temperature * temperature)


class SafeGradProjector:
    """
    SafeGrad Gradient Surgery Context.

    Intercepts the backward pass of a user-task loss and projects the resulting
    gradients orthogonally to a pre-computed safety gradient.  The conflict test
    and projection operate on the *global* concatenated gradient vector (all
    trainable parameters flattened into one vector), matching Algorithm 1 of the
    paper (Eq. 3-4).  Per-tensor tests would be incorrect: a parameter that has a
    locally negative dot product with the safety gradient may still be part of a
    globally non-conflicting update, and vice versa.

    Usage:
```python
        projector = SafeGradProjector(model)

        # 1. Compute and store safety gradients
        model.zero_grad()
        safety_loss = compute_safety_loss(model, safety_batch)
        projector.capture_safety_gradients(safety_loss)

        # 2. Compute user loss, then trigger global surgery in __exit__
        model.zero_grad()
        user_loss = compute_user_loss(model, user_batch)

        with projector.apply_gradient_surgery():
            user_loss.backward()
            # __exit__ runs here and performs the global projection

        optimizer.step()
```
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.safety_grads: Dict[int, torch.Tensor] = {}
        self._hooks: list = []

    def capture_safety_gradients(self, safety_loss: torch.Tensor) -> None:
        """
        Backpropagates the given safety loss and stores the exact gradient vector
        for each trainable parameter. Gradients on the model are then zeroed.
        """
        if not safety_loss.requires_grad:
            logger.warning("SafeGrad: Provided safety_loss does not require gradients. Skipping capture.")
            return

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        grads = torch.autograd.grad(
            safety_loss,
            trainable_params,
            allow_unused=True,
            retain_graph=False
        )

        self.safety_grads = {}
        for param, grad in zip(trainable_params, grads):
            if grad is not None:
                self.safety_grads[id(param)] = grad.detach().clone()

    class _ContextManager:
        """Context manager for global gradient surgery (paper Algorithm 1, Eq. 3-4).

        The conflict test and projection must operate on the *global* concatenated
        gradient vector, not independently per tensor.  Per-tensor tests can fire
        incorrectly (local negative dot product when global is positive, or miss
        a global conflict when all individual tensors look fine).

        We do the surgery in __exit__ after backward() has fully populated all
        .grad attributes, then scatter the corrected global vector back into each
        parameter's .grad so the downstream optimizer steps on the safe update.
        """

        def __init__(self, projector: "SafeGradProjector"):
            self.projector = projector

        def __enter__(self):
            if not self.projector.safety_grads:
                logger.warning("SafeGrad: No safety gradients captured. Surgery will not be applied.")
            # No per-param hooks needed — surgery is done globally in __exit__
            # after backward() has finished populating all .grad attributes.
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Don't touch gradients if backward raised an exception.
            if exc_type is not None:
                return
            if not self.projector.safety_grads:
                return

            # Collect params that have both a stored safety gradient
            # and a populated .grad from the user backward pass.
            params = [
                p for p in self.projector.model.parameters()
                if p.requires_grad and id(p) in self.projector.safety_grads
            ]
            if not params:
                return

            # Flatten all user gradients and alignment gradients into two
            # global vectors.  Params without a .grad get a zero contribution.
            # Record which params actually had a user gradient: params whose grad
            # was None are only padded with zeros for the global conflict test and
            # must NOT receive a synthesized gradient during scatter-back, otherwise
            # optimizer.step() would update parameters the user loss never touched.
            flat_user, flat_align = [], []
            had_user_grad = {}
            for p in params:
                had_user_grad[id(p)] = p.grad is not None
                gu = p.grad if p.grad is not None else torch.zeros_like(p)
                ga = self.projector.safety_grads[id(p)].to(gu.device, gu.dtype)
                flat_user.append(gu.reshape(-1))
                flat_align.append(ga.reshape(-1))

            g_user = torch.cat(flat_user)
            g_align = torch.cat(flat_align)

            # Single global conflict test (paper Eq. 3).
            # Use elementwise multiply + sum rather than torch.dot to avoid
            # the int32 length overflow on models with more than ~2B parameters.
            dot = (g_user * g_align).sum()
            align_sq = (g_align * g_align).sum() + 1e-12

            if dot < 0:
                # Subtract the conflicting component (paper Eq. 4).
                g_user = g_user - (dot / align_sq) * g_align

            # Scatter the corrected global vector back into each parameter's .grad.
            offset = 0
            for p in params:
                n = p.numel()
                chunk = g_user[offset: offset + n].reshape(p.shape)
                # Only modify grads that genuinely existed. Skip params whose
                # original user grad was None (leave .grad None) so we never
                # create an update for an untouched parameter.
                if had_user_grad[id(p)]:
                    p.grad.copy_(chunk)
                offset += n

    def apply_gradient_surgery(self) -> _ContextManager:
        """
        Returns a context manager.  Call ``user_loss.backward()`` inside this
        context; the global gradient surgery runs automatically in ``__exit__``
        after backward completes.
        """
        return self._ContextManager(self)
