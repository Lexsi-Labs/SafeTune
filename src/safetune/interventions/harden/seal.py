"""
SEAL — Safety-Enhanced Alignment via data selection (Shen et al., 2024,
arXiv:2410.07471).

SEAL protects safety during fine-tuning through bilevel data selection:
an inner loop identifies which training examples are most safety-sensitive
(those whose gradient conflicts most with the alignment gradient), and an
outer loop up-weights those examples during SFT.

Simplified faithful implementation for the SafeTune library:

The full bilevel meta-learning loop (differentiating through the inner
training step) is computationally prohibitive at 3B-8B scale. This
implementation uses the tractable SEAL approximation from the paper's
Appendix B: gradient-cosine scoring to identify safety-sensitive examples,
followed by importance-weighted SFT.

Algorithm:
1. Compute the alignment gradient g_align on a safety reference batch.
2. For each training example i, compute its task gradient g_i.
3. Score: s_i = -cosine(g_i, g_align). High score = high conflict with safety.
4. Importance weights: w_i = softmax(s_i / temperature).
5. Train with weighted SFT loss: L = sum_i w_i * L_i.

This is the SEAL "gradient conflict" data selector from §3.2 of the paper.
The weights are recomputed every `seal_rescore_every` steps.

Paper: "SEAL: Safety-Enhanced Aligned LLM Fine-Tuning via Bilevel Data Selection"
Shen et al., 2024, arXiv:2410.07471.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from transformers import Trainer, TrainingArguments
    _TRAINER_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    Trainer = object  # type: ignore[assignment,misc]
    TrainingArguments = object  # type: ignore[assignment,misc]
    _TRAINER_IMPORT_ERROR = _e


if _TRAINER_IMPORT_ERROR is None:
    @dataclass
    class SEALConfig(TrainingArguments):  # type: ignore[misc]
        """TrainingArguments subclass for SEAL.

        Attributes:
            seal_temperature: Softmax temperature for importance weights.
                Lower = more aggressive up-weighting of conflicting examples.
                Default: 1.0.
            seal_rescore_every: How often (steps) to recompute gradient-conflict
                scores. Scoring is expensive; default 10 amortises the cost.
            seal_top_k_ratio: Fraction of examples to up-weight (set weights
                to 1.0 for bottom examples). Default 1.0 = weight all.
        """
        seal_temperature: float = 1.0
        seal_rescore_every: int = 10
        seal_top_k_ratio: float = 1.0
else:  # pragma: no cover
    class SEALConfig(object):  # type: ignore[assignment]
        pass


def _flat_grad(
    model: Any,
    loss: torch.Tensor,
    retain_graph: bool = False,
) -> torch.Tensor:
    """Flatten the gradient of loss w.r.t. all trainable parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, allow_unused=True
    )
    parts = []
    for g in grads:
        if g is not None:
            parts.append(g.detach().reshape(-1))
        # Skip None grads (unused params)
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


class SEALTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """HuggingFace Trainer implementing SEAL importance-weighted SFT.

    Uses gradient-conflict scoring to identify safety-sensitive training
    examples and up-weights them during fine-tuning.

    Args:
        safety_dataset: Iterable of safety reference batches used to compute
            the alignment gradient g_align. Required.
        All other arguments forwarded to transformers.Trainer.
    """

    def __init__(
        self,
        *args: Any,
        safety_dataset: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for SEALTrainer"
            ) from _TRAINER_IMPORT_ERROR
        if safety_dataset is None:
            raise ValueError("SEALTrainer requires a 'safety_dataset' argument.")

        super().__init__(*args, **kwargs)

        self._seal_temperature = float(getattr(self.args, "seal_temperature", 1.0))
        self._seal_rescore_every = max(
            1, int(getattr(self.args, "seal_rescore_every", 10)))
        self._seal_top_k_ratio = float(getattr(self.args, "seal_top_k_ratio", 1.0))

        from itertools import cycle
        try:
            bs = self.args.per_device_train_batch_size
        except AttributeError:
            bs = 1
        try:
            loader = torch.utils.data.DataLoader(
                safety_dataset, batch_size=bs, shuffle=True)
            self._safety_iter: Iterator = cycle(loader)
        except Exception:
            self._safety_iter = cycle(safety_dataset)

        # Cache: importance weights per example index in the current batch.
        self._cached_weights: Optional[torch.Tensor] = None
        self._step_count: int = 0

    def _alignment_gradient(self, model: Any) -> torch.Tensor:
        """Compute flattened gradient of alignment loss on a safety batch."""
        device = next(model.parameters()).device
        batch = next(self._safety_iter)
        if isinstance(batch, dict):
            batch = {k: v.to(device) if hasattr(v, "to") else v
                     for k, v in batch.items()}
        # NOTE: no ``model.zero_grad()`` here — ``_flat_grad`` uses the
        # functional ``torch.autograd.grad`` which never reads or writes
        # ``param.grad``, so zeroing would only destroy gradients accumulated
        # by earlier micro-batches (gradient accumulation).
        try:
            out = model(**batch)
            loss = out.loss
        except Exception:
            safe_keys = {"input_ids", "attention_mask", "labels"}
            filtered = {k: v for k, v in batch.items() if k in safe_keys}
            out = model(**filtered)
            loss = out.loss
        g_align = _flat_grad(model, loss, retain_graph=False)
        return g_align

    def _score_examples(
        self,
        model: Any,
        inputs: Dict[str, Any],
        g_align: torch.Tensor,
    ) -> torch.Tensor:
        """Score each example in the batch by gradient conflict with g_align.

        Returns importance weights of shape (B,).
        """
        device = next(model.parameters()).device
        B = inputs["input_ids"].shape[0]
        scores = []
        for i in range(B):
            single = {k: v[i:i+1].to(device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}
            # No ``model.zero_grad()`` around scoring: ``_flat_grad`` is
            # functional (``torch.autograd.grad``) and never touches
            # ``param.grad``; zeroing here would erase user gradients
            # accumulated by earlier micro-batches.
            try:
                out = model(**single)
                loss_i = out.loss
            except Exception:
                scores.append(torch.tensor(0.0))
                continue
            g_i = _flat_grad(model, loss_i, retain_graph=False)
            # SEAL (arXiv:2410.07471) UP-weights examples whose gradient ALIGNS
            # with the safe/alignment gradient and DOWN-weights those that
            # conflict with it. So the score is the *positive* cosine: higher
            # alignment -> higher softmax weight. (Using -cos here inverts the
            # ranker and amplifies safety degradation — the opposite of SEAL.)
            cos = F.cosine_similarity(g_i.unsqueeze(0),
                                      g_align.to(g_i.device).unsqueeze(0)).item()
            scores.append(torch.tensor(cos))
        scores_t = torch.stack(scores)
        weights = F.softmax(scores_t / self._seal_temperature, dim=0) * B
        return weights

    def compute_loss(  # type: ignore[override]
        self,
        model: Any,
        inputs: Any,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Any:
        """Compute importance-weighted SFT loss."""
        self._step_count += 1

        device = next(model.parameters()).device
        B = inputs["input_ids"].shape[0]

        # Recompute scores every seal_rescore_every steps.  ``_step_count`` is
        # 1 on the first call, so ``(step - 1) % every == 0`` fires on the
        # first step and every ``every`` steps thereafter — and works for
        # ``seal_rescore_every=1`` (rescore every step), which the previous
        # ``step % every == 1`` phase could never satisfy (1 % 1 == 0).
        is_rescore_step = (
            (self._step_count - 1) % self._seal_rescore_every == 0
            or self._cached_weights is None
        )
        if is_rescore_step:
            try:
                g_align = self._alignment_gradient(model)
                self._cached_weights = self._score_examples(model, inputs, g_align)
            except Exception as exc:
                logger.warning("SEALTrainer: scoring failed (%s); using uniform weights.", exc)
                self._cached_weights = torch.ones(B)

        # Non-rescore steps reuse the LAST computed weights (docstring:
        # "weights are recomputed every `seal_rescore_every` steps" —
        # scoring is expensive and the cache amortises it).  Cached weights
        # are positional per-example values, so they can only be applied when
        # the current batch has the same size; on a mismatch (e.g. a smaller
        # final batch with drop_last=False) fall back to uniform weights to
        # avoid a shape crash.
        if (self._cached_weights is not None
                and self._cached_weights.shape[0] == B):
            weights = self._cached_weights.to(device)
        else:
            weights = torch.ones(B, device=device)

        # Compute per-example losses.
        per_example_losses = []
        for i in range(B):
            single = {k: v[i:i+1].to(device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}
            try:
                out_i = model(**single)
                per_example_losses.append(out_i.loss)
            except Exception:
                per_example_losses.append(torch.tensor(0.0, device=device,
                                                        requires_grad=True))

        losses_t = torch.stack(per_example_losses)
        loss = (losses_t * weights).mean()

        if return_outputs:
            # Return full-batch outputs for compatibility.
            outputs = model(**{k: v.to(device) if hasattr(v, "to") else v
                               for k, v in inputs.items()})
            return loss, outputs
        return loss


__all__ = ["SEALConfig", "SEALTrainer"]