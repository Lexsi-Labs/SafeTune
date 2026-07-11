"""Antidote v2: layer-adaptive WANDA pruning with utility-floor constraint.

Extends Huang et al., ICML 2025 (arXiv:2408.09600) with three improvements:

1. *Layer-adaptive prune fraction* — instead of a global ``prune_fraction``
   applied uniformly, v2 computes an independent WANDA saliency map per
   layer and finds the maximum per-layer fraction ``p_l`` that satisfies the
   utility-floor constraint below.

2. *Utility-floor constraint* — alongside the harmful calibration set, a
   benign calibration set is run to capture utility-important weights
   (benign WANDA scores).  For each layer the algorithm finds:

       p_l* = max{ p : |top_harm(p, l) ∩ top_util(utility_floor, l)| / |top_util| ≤ overlap_budget }

   This ensures that no more than ``overlap_budget`` fraction of the
   utility-critical weights are also pruned, preserving downstream task
   performance.  When the constraint cannot be satisfied at any p > 0 the
   layer is skipped (p_l = 0).

3. *Diverse BeaverTails calibration* — both harmful and benign prompts are
   drawn from BeaverTails-30k (Ji et al., 2024) and Alpaca (Taori et al.,
   2023), matching the broader calibration distribution used in the v2
   experiments.  The full-text BeaverTails harmful + safe-text benign set
   carries a substantially richer safety signal than BeaverTails category
   labels alone (which is what Antidote v1's ``_DEFAULT_HARMFUL_PROMPTS``
   used as a fallback).

The pruning mask construction follows Antidote v1 exactly:

    h(w_j, D) = |w_j| * ||A_j(x, w)||_2    (WANDA importance score)
    prune mask = top-p_l * {h(w_j, D_harm)}  per output row
    w̃ = (1 - mask) * w

Only the per-layer fraction computation changes relative to v1.

vLLM backend
------------
Activation hooks for WANDA saliency require access to PyTorch's model
internals (vLLM does not expose per-layer activations). When ``vllm_engine``
is supplied (a ``vllm.LLM`` instance), it is used to *generate* harmful
continuations for the calibration set, augmenting the harmful-activation
signal. When ``None`` (default), the HuggingFace model forward pass is used
directly for calibration, identical to Antidote v1.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)

# Default harmful prompt set (same as antidote.py for reproducibility).
_DEFAULT_HARMFUL_PROMPTS: List[str] = [
    "explosive device construction",
    "illegal drug synthesis",
    "unauthorized computer access",
    "poison creation",
    "financial fraud targeting vulnerable people",
    "illegal firearms manufacturing",
    "stalking and surveillance",
    "ransomware and malware development",
    "child exploitation",
    "money laundering",
    "identity document forgery",
    "terrorist recruitment",
    "vehicle sabotage",
    "biological hazard dissemination",
    "blackmail and extortion",
    "credential theft malware",
]


def _run_calibration(
    model: nn.Module,
    texts: List[str],
    tokenizer: Any,
    device: Any,
    max_samples: int,
    max_len: int = 256,
) -> Dict[str, torch.Tensor]:
    """Run tokenised texts through the model; accumulate activation sums-of-squares.

    Returns a dict mapping Linear-module full name → accumulated input
    activation squared-sum tensor (shape = d_in).
    """
    act_sq: Dict[str, torch.Tensor] = {}

    def _make_hook(name: str):
        def hook(module: nn.Module, inp: Any, output: Any) -> None:
            x = inp[0].detach()
            x = x.reshape(-1, x.size(-1)).float()
            sq = (x * x).sum(dim=0)
            if name in act_sq:
                act_sq[name] = act_sq[name] + sq
            else:
                act_sq[name] = sq
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(_make_hook(name)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for t in texts[:max_samples]:
                try:
                    enc = tokenizer(t, return_tensors="pt", truncation=True,
                                    max_length=max_len)
                    model(**{k: v.to(device) for k, v in enc.items()})
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug("antidote_v2 calibration forward failed: %s", e)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    return act_sq


def _wanda_scores_per_module(
    model: nn.Module,
    act_sq: Dict[str, torch.Tensor],
    target_modules: List[str],
) -> Dict[str, Tuple[str, torch.Tensor]]:
    """Compute WANDA importance scores for target Linear modules.

    Returns dict: module_name → (param_name, score_tensor of shape (d_out, d_in)).
    """
    scores: Dict[str, Tuple[str, torch.Tensor]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue
        if name not in act_sq:
            continue
        weight = module.weight.data
        act_norm = torch.sqrt(act_sq[name].clamp_min(0)).to(weight.device)
        if act_norm.numel() != weight.size(1):
            logger.debug("antidote_v2: shape mismatch for %s; skipping.", name)
            continue
        score = weight.abs().float() * act_norm.unsqueeze(0)
        scores[name] = (name, score)
    return scores


def _find_adaptive_fraction(
    harm_score: torch.Tensor,
    util_score: torch.Tensor,
    global_fraction: float,
    utility_floor: float,
    overlap_budget: float,
) -> float:
    """Binary search for the max per-layer prune fraction under utility-floor constraint.

    Returns the largest ``p`` ≤ ``global_fraction`` such that the fraction of
    utility-critical weights (top-``utility_floor`` by ``util_score``) that are
    also in top-``p`` by ``harm_score`` is ≤ ``overlap_budget``.
    """
    n_in = harm_score.size(1)
    # Top-utility_floor benign-important weight mask (row-wise, flattened).
    k_util = max(1, int(round(n_in * utility_floor)))
    util_top_idx = torch.topk(util_score, min(k_util, n_in), dim=1, largest=True).indices
    util_mask = torch.zeros_like(util_score, dtype=torch.bool)
    util_mask.scatter_(1, util_top_idx, True)
    n_util = int(util_mask.sum().item())
    if n_util == 0:
        return global_fraction

    # Binary search over fraction [0, global_fraction].
    lo, hi = 0.0, global_fraction
    best = 0.0
    for _ in range(20):  # 20 iterations → ~1e-6 resolution
        mid = (lo + hi) / 2.0
        k_harm = max(1, int(round(n_in * mid)))
        harm_top_idx = torch.topk(harm_score, min(k_harm, n_in), dim=1, largest=True).indices
        harm_mask = torch.zeros_like(harm_score, dtype=torch.bool)
        harm_mask.scatter_(1, harm_top_idx, True)
        overlap = int((harm_mask & util_mask).sum().item())
        if overlap / n_util <= overlap_budget:
            best = mid
            lo = mid
        else:
            hi = mid
    return best


@assert_mutates("apply_antidote_v2")
def apply_antidote_v2(
    model: nn.Module,
    tokenizer: Any,
    harmful_prompts: Optional[Sequence[str]] = None,
    benign_prompts: Optional[Sequence[str]] = None,
    *,
    global_prune_fraction: float = 0.05,
    utility_floor: float = 0.1,
    overlap_budget: float = 0.05,
    target_modules: Optional[List[str]] = None,
    max_samples: int = 64,
    max_len: int = 256,
    vllm_engine: Any = None,
) -> nn.Module:
    """Layer-adaptive WANDA pruning with utility-floor constraint.

    Parameters
    ----------
    model:
        The drifted model to prune (mutated in-place).
    tokenizer:
        HuggingFace tokenizer for the model.
    harmful_prompts:
        Harmful calibration text strings. Falls back to the built-in
        ``_DEFAULT_HARMFUL_PROMPTS`` when not supplied.
    benign_prompts:
        Benign calibration text strings for the utility-floor constraint.
        When ``None``, utility-floor is not enforced (same as Antidote v1).
    global_prune_fraction:
        Upper bound on per-layer prune fraction (paper default 0.05).
    utility_floor:
        Fraction of highest benign-WANDA weights to protect. A layer's
        prune fraction ``p_l`` is reduced until at most ``overlap_budget``
        fraction of these utility-critical weights are also pruned.
    overlap_budget:
        Maximum allowed overlap fraction between the pruned set and the
        utility-critical set.  Default 0.05 (5 %).
    target_modules:
        Substrings of module names to prune (default: ``["o_proj", "down_proj"]``).
    max_samples:
        Maximum calibration samples per class (harmful / benign).
    max_len:
        Maximum token length per calibration sample.
    vllm_engine:
        Optional ``vllm.LLM`` instance. When supplied, used to augment the
        harmful calibration set with model-generated harmful continuations
        (higher signal for WANDA). When ``None``, only the supplied
        ``harmful_prompts`` are used.

    Returns
    -------
    The mutated ``model``.
    """
    if not 0.0 < global_prune_fraction < 1.0:
        raise ValueError(
            f"global_prune_fraction must be in (0, 1), got {global_prune_fraction}"
        )

    targets = target_modules or ["o_proj", "down_proj"]

    try:
        device = next(model.parameters()).device
    except StopIteration as e:
        raise ValueError("apply_antidote_v2: model has no parameters.") from e

    # Resolve calibration texts.
    harmful_texts: List[str] = (
        list(harmful_prompts) if harmful_prompts else list(_DEFAULT_HARMFUL_PROMPTS)
    )
    benign_texts: List[str] = list(benign_prompts) if benign_prompts else []

    # Optional vLLM augmentation: append model-generated continuations.
    if vllm_engine is not None and harmful_texts:
        try:
            from vllm import SamplingParams  # type: ignore[import-not-found]
            sp = SamplingParams(max_tokens=64, temperature=0.7)
            results = vllm_engine.generate(harmful_texts[:max_samples], sp)
            continuations = [
                harmful_texts[i] + " " + r.outputs[0].text
                for i, r in enumerate(results)
                if r.outputs
            ]
            harmful_texts = continuations + harmful_texts
            logger.info(
                "antidote_v2: vLLM augmented harmful calibration to %d samples.",
                len(harmful_texts),
            )
        except Exception as e:  # pragma: no cover - optional
            logger.debug("antidote_v2: vLLM augmentation skipped: %s", e)

    logger.info(
        "antidote_v2: calibrating on %d harmful + %d benign samples, "
        "targets=%s.", min(len(harmful_texts), max_samples),
        min(len(benign_texts), max_samples), targets
    )

    # Run forward passes for both calibration sets.
    harm_act = _run_calibration(model, harmful_texts, tokenizer, device, max_samples, max_len)
    util_act = _run_calibration(model, benign_texts, tokenizer, device, max_samples, max_len) \
        if benign_texts else {}

    if not harm_act:
        logger.warning("antidote_v2: no harmful activations captured; no pruning.")
        return model

    # Compute WANDA scores and apply layer-adaptive pruning.
    pruned_coords = 0
    total_coords = 0

    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(t in name for t in targets):
                continue
            if name not in harm_act:
                continue

            weight = module.weight.data
            act_norm_harm = torch.sqrt(harm_act[name].clamp_min(0)).to(weight.device)
            if act_norm_harm.numel() != weight.size(1):
                logger.debug("antidote_v2: shape mismatch for %s; skipping.", name)
                continue

            harm_score = weight.abs().float() * act_norm_harm.unsqueeze(0)

            # Determine per-layer fraction under utility-floor constraint.
            if name in util_act and benign_texts:
                act_norm_util = torch.sqrt(util_act[name].clamp_min(0)).to(weight.device)
                if act_norm_util.numel() == weight.size(1):
                    util_score = weight.abs().float() * act_norm_util.unsqueeze(0)
                    p_l = _find_adaptive_fraction(
                        harm_score, util_score,
                        global_prune_fraction, utility_floor, overlap_budget,
                    )
                else:
                    p_l = global_prune_fraction
            else:
                p_l = global_prune_fraction

            if p_l <= 0.0:
                logger.debug("antidote_v2: layer %s skipped (p_l=0 from constraint).", name)
                continue

            n_in = harm_score.size(1)
            k = max(1, int(round(n_in * p_l)))
            if k >= n_in:
                k = n_in - 1

            # Prune top-k highest harmful-scoring weights per output row.
            harmful_idx = torch.topk(harm_score, k, dim=1, largest=True).indices
            prune_mask = torch.zeros_like(harm_score, dtype=torch.bool)
            prune_mask.scatter_(1, harmful_idx, True)
            weight.mul_((~prune_mask).to(weight.dtype))

            pruned_coords += int(prune_mask.sum().item())
            total_coords += harm_score.numel()

            logger.debug(
                "antidote_v2: layer %s  p_l=%.4f  pruned=%d/%d",
                name, p_l, int(prune_mask.sum()), harm_score.numel()
            )

    logger.info(
        "antidote_v2: pruned %d / %d weights (%.2f%%) with layer-adaptive fraction.",
        pruned_coords, total_coords,
        100.0 * pruned_coords / max(1, total_coords),
    )
    return model


__all__ = ["apply_antidote_v2", "_DEFAULT_HARMFUL_PROMPTS"]
