"""
TracIn-style influence approximation (Pruthi et al., NeurIPS 2020).

Reference: "Estimating Training Data Influence by Tracing Gradient Descent",
Pruthi, Liu, Kale, Sundararajan, NeurIPS 2020, arXiv:2002.08484.
Reference implementation: https://github.com/frederick0329/TracIn

TracIn approximates the influence of a training example ``z`` on a test
example ``z'`` by summing, **over the training checkpoints that were saved
during training**, the dot product of their per-example loss gradients,
scaled by the learning rate that was in effect at that checkpoint:

    TracInCP(z, z') = Sum_t  eta_t * grad L(z; theta_t) . grad L(z'; theta_t)

The multi-checkpoint sum is the defining mechanism of the method (it traces
the cumulative effect of SGD across training). ``TracInCP`` uses the saved
checkpoints; the idealised version integrates along the whole trajectory.

Sign convention (matches the paper / reference repo):

* A **positive** influence means ``z`` is a *proponent* of ``z'`` -- a
  gradient step on ``z`` *reduces* the loss on ``z'``. This is because the
  SGD update ``theta <- theta - eta * grad L(z)`` changes the test loss by
  approximately ``-eta * grad L(z) . grad L(z')``; the reduction in test loss
  is therefore proportional to ``+eta * grad L(z) . grad L(z')``.
* A **negative** influence means ``z`` is an *opponent* of ``z'`` -- a step
  on ``z`` *increases* the loss on ``z'``.

For SafeTune we use TracIn to identify, for a refusal-style test point, the
training examples that most affect it. In an unlearning context the examples
worth scrubbing are the ones with large positive influence on a *harmful*
test point (proponents of harm), or large negative influence on a refusal
test point (opponents of refusal) -- in both cases the sign must be read
according to the convention above, not inverted.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _flat_grads(model: nn.Module) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            parts.append(p.grad.detach().reshape(-1).float())
    return torch.cat(parts) if parts else torch.zeros(0)


def _example_gradient(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model, batch)
    loss.backward()
    g = _flat_grads(model)
    model.zero_grad(set_to_none=True)
    return g


def tracin_influence(
    model: nn.Module,
    train_examples: Iterable[Dict[str, torch.Tensor]],
    test_example: Dict[str, torch.Tensor],
    loss_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
    checkpoints: Optional[Sequence[nn.Module]] = None,
    learning_rates: Optional[Sequence[float]] = None,
) -> List[float]:
    """Approximate the influence of each training example on the test example.

    Implements the TracInCP estimator (Pruthi et al., NeurIPS 2020):

        TracInCP(z, z') = Sum_t  eta_t * grad L(z; theta_t) . grad L(z'; theta_t)

    Parameters
    ----------
    model:
        A model used as a checkpoint. If ``checkpoints`` is given this argument
        is ignored for scoring and only the checkpoint list is used.
    train_examples:
        Iterable of training batches to score.
    test_example:
        The single test batch whose loss we trace.
    loss_fn:
        ``loss_fn(model, batch) -> scalar tensor`` used for both train and test
        gradients.
    checkpoints:
        Optional ordered sequence of model checkpoints saved during training.
        TracIn sums the gradient dot product over **all** of these. When this
        is ``None`` (or has length 1) the estimator degrades to
        *TracInCP-single-point* -- a single-checkpoint dot product, which is
        only a crude proxy for the full trajectory sum and is provided for
        convenience / cheap scoring.
    learning_rates:
        Optional per-checkpoint learning rates ``eta_t``. Must match the number
        of checkpoints actually used. If ``None`` every checkpoint is weighted
        by ``1.0`` (unweighted sum). The paper weights each checkpoint by the
        learning rate in effect when that checkpoint was saved.

    Returns
    -------
    list of float
        One scalar influence per training example.

    Sign convention
    ---------------
    A **positive** value means the training example is a *proponent* of the
    test example: a gradient step on it *reduces* the test loss. A **negative**
    value means it is an *opponent* and *increases* the test loss. This matches
    Pruthi et al. and the reference implementation
    (https://github.com/frederick0329/TracIn) -- proponents have positive
    scores proportional to loss reduction.
    """
    # Resolve the set of checkpoints to trace over.
    ckpts: Sequence[nn.Module]
    if checkpoints is None or len(checkpoints) == 0:
        ckpts = [model]
    else:
        ckpts = list(checkpoints)

    # Resolve per-checkpoint learning-rate weights eta_t.
    if learning_rates is None:
        etas: List[float] = [1.0] * len(ckpts)
    else:
        etas = [float(x) for x in learning_rates]
        if len(etas) != len(ckpts):
            raise ValueError(
                "learning_rates must have one entry per checkpoint: "
                f"got {len(etas)} rates for {len(ckpts)} checkpoint(s)."
            )

    if len(ckpts) == 1:
        logger.info(
            "tracin_influence: single checkpoint -> degrading to "
            "TracInCP-single-point (no trajectory sum)."
        )

    # train_examples may be a one-shot iterator; materialise so we can reuse it
    # across every checkpoint.
    train_list: List[Dict[str, torch.Tensor]] = list(train_examples)
    out: List[float] = [0.0] * len(train_list)

    for ckpt, eta in zip(ckpts, etas):
        test_grad = _example_gradient(ckpt, test_example, loss_fn)
        for i, ex in enumerate(train_list):
            g = _example_gradient(ckpt, ex, loss_fn)
            if (
                g.numel() == 0
                or test_grad.numel() == 0
                or g.shape != test_grad.shape
            ):
                continue
            # TracIn term for this checkpoint: eta_t * grad(z) . grad(z').
            out[i] += eta * float(g.dot(test_grad).item())

    logger.info(
        "tracin_influence: scored %d training examples over %d checkpoint(s).",
        len(out),
        len(ckpts),
    )
    return out


__all__ = ["tracin_influence"]
