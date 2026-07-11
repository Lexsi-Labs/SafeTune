"""
SEAL: Safety-Enhanced Aligned LLM Fine-tuning via Bilevel Data Selection.
arXiv 2024 — hanshen95/SEAL

SEAL fine-tuning:
1. Trains a data selector (inner loop) via bilevel optimization that assigns 
   importance weights to fine-tuning examples based on their safety impact.
2. Filters the fine-tuning dataset using hard-thresholding on the learned weights.
3. Fine-tunes the LLM only on the curated, safe-leaning subset.

This module provides:
- SEALDataSelector: gradient-based importance scorer using a proxy bilevel step.
- select_safe_dataset: convenience function to filter a list of examples.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SEALConfig:
    """Configuration for SEAL bilevel data selection."""
    # Fraction of data to keep after filtering (e.g., 0.8 = keep top-80% safest)
    keep_ratio: float = 0.8
    # Number of inner-loop bilevel steps to estimate importance
    inner_steps: int = 5
    # Learning rate for the inner loop meta gradient
    inner_lr: float = 1e-4
    # Safety anchor dataset size used for the validation signal
    safety_anchor_size: int = 64
    # Seed for reproducibility in candidate scoring
    seed: int = 42


class SEALDataSelector:
    """
    Gradient-based data selector using a bilevel proxy objective.

    The key idea: an example is "safe" if fine-tuning ON it does NOT degrade
    the model's performance on a held-out safety anchor set.
    We approximate this by computing the gradient alignment between the example's
    gradient and the safety anchor gradient — examples with high alignment are kept.
    """

    def __init__(
        self,
        model: Any,
        config: Optional[SEALConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or SEALConfig()
        self._importance_scores: List[float] = []

    def score_examples(
        self,
        examples: List[Dict[str, Any]],
        compute_example_loss: Callable[[Any, Dict[str, Any]], Any],
        compute_safety_anchor_loss: Callable[[Any], Any],
    ) -> List[float]:
        """
        Assign an importance score to each example.

        Higher score = safer to include.
        
        Args:
            examples: List of fine-tuning examples (as dicts).
            compute_example_loss: callable(model, example) -> scalar loss.
            compute_safety_anchor_loss: callable(model) -> scalar safety loss on anchor set.

        Returns:
            List of float importance scores, one per example.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("SEAL requires PyTorch.")

        # Compute safety anchor gradient once
        self.model.zero_grad()
        anchor_loss = compute_safety_anchor_loss(self.model)
        anchor_loss.backward()
        anchor_grads = {
            n: p.grad.detach().clone()
            for n, p in self.model.named_parameters()
            if p.grad is not None
        }
        self.model.zero_grad()

        scores = []
        for i, example in enumerate(examples):
            self.model.zero_grad()
            try:
                ex_loss = compute_example_loss(self.model, example)
                ex_loss.backward()
                ex_grads = {
                    n: p.grad.detach().clone()
                    for n, p in self.model.named_parameters()
                    if p.grad is not None
                }
                self.model.zero_grad()

                # Score = cosine similarity between example gradient and anchor gradient
                # (High similarity = example points model in same direction as safety anchor)
                dot, norm_ex, norm_anc = 0.0, 0.0, 0.0
                for n in ex_grads:
                    if n in anchor_grads:
                        eg = ex_grads[n].view(-1).float()
                        ag = anchor_grads[n].view(-1).float()
                        dot += torch.dot(eg, ag).item()
                        norm_ex += eg.norm().item() ** 2
                        norm_anc += ag.norm().item() ** 2

                if norm_ex > 0 and norm_anc > 0:
                    score = dot / ((norm_ex ** 0.5) * (norm_anc ** 0.5))
                else:
                    score = 0.0

            except Exception as e:
                logger.warning("SEAL: error scoring example %d: %s", i, e)
                score = 0.0

            scores.append(float(score))

        self._importance_scores = scores
        logger.info(
            "SEAL scored %d examples. Mean score: %.4f", len(scores), sum(scores) / max(len(scores), 1)
        )
        return scores

    def filter_dataset(
        self,
        examples: List[Dict[str, Any]],
        scores: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hard-threshold filter: keep the top `keep_ratio` fraction by importance score.
        """
        sc = scores if scores is not None else self._importance_scores
        if not sc:
            logger.warning("SEAL: no scores available. Returning full dataset.")
            return examples

        n_keep = max(1, int(len(examples) * self.config.keep_ratio))
        indexed = sorted(enumerate(sc), key=lambda x: x[1], reverse=True)
        keep_indices = {i for i, _ in indexed[:n_keep]}
        retained = [ex for i, ex in enumerate(examples) if i in keep_indices]

        logger.info(
            "SEAL filtered %d -> %d examples (keep_ratio=%.2f).",
            len(examples), len(retained), self.config.keep_ratio,
        )
        return retained


def select_safe_dataset(
    model: Any,
    examples: List[Dict[str, Any]],
    compute_example_loss: Callable[[Any, Dict[str, Any]], Any],
    compute_safety_anchor_loss: Callable[[Any], Any],
    config: Optional[SEALConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function: score + filter in one call.
    """
    selector = SEALDataSelector(model=model, config=config)
    scores = selector.score_examples(examples, compute_example_loss, compute_safety_anchor_loss)
    return selector.filter_dataset(examples, scores)
