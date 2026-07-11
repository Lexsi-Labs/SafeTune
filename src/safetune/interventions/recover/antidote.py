"""Antidote: WANDA-based pruning of safety-critical (harmful) weights.

Faithful re-implementation of the one-shot pruning stage of

    Huang et al., "Antidote: Post-fine-tuning Safety Alignment for Large
    Language Models against Harmful Fine-tuning Attack", ICML 2025,
    arXiv:2408.09600.
    Unofficial repo: https://github.com/git-disl/Antidote

Algorithm (paper, Sec. 4 / Eq. for ``h`` and ``m``):

1. Run a *harmful realignment dataset* ``D_realign`` (harmful prompt + harmful
   answer pairs, sampled from BeaverTails in the paper) through the model and
   accumulate, per weight coordinate, the L2 norm of the input activations.
2. Compute the WANDA importance score
   ``h(w, D_realign)_j = (1/|D|) * |w_j| * ||A_j(x, w)||_2``  -- weight
   magnitude AND activation norm.
3. One-shot prune the **top-alpha highest-scoring** weights (``m =
   ArgTopK_alpha(h)``) -- these are the weights most responsible for harmful
   generations -- via ``w_tilde = (1 - m) * w``.

This module previously delegated to ``core.patches.antidote.AntidotePatch``,
which (a) inverted the keep/prune mask -- it pruned everything *below* the
top-k threshold, destroying ~95% of each layer at the default
``prune_fraction=0.05``; (b) calibrated the WANDA score with a
``torch.randn`` dummy tensor, so the score carried no safety signal; and
(c) referenced an undefined ``logger``. The pruning is now performed here
directly with those bugs fixed.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import torch
import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)

# A small built-in harmful realignment prompt set used to calibrate the WANDA
# activation norms when the caller does not supply ``harmful_prompts`` /
# ``dataset``.  The strings below are **harm-category labels** drawn from
# BeaverTails (Ji et al., 2024) — they name the harm category, not spell out
# instructions.  For production use, pass a tokenized BeaverTails or HarmBench
# slice directly via the ``dataset=`` or ``harmful_prompts=`` arguments.
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


def _build_realignment_inputs(
    model: nn.Module,
    harmful_prompts: Optional[Sequence[str]],
    dataset: Optional[Sequence[Any]],
    tokenizer: Any,
    device: Any,
    max_samples: int,
) -> List[Any]:
    """Build forward-pass inputs from a real harmful realignment dataset.

    Returns a list of input dicts/tensors ready to feed to ``model(...)``.
    Falls back to the built-in harmful prompt set when nothing is supplied.
    """
    # 1. Resolve the harmful texts.
    texts: List[str] = []
    if dataset is not None:
        for row in dataset:
            if isinstance(row, str):
                texts.append(row)
            elif isinstance(row, dict):
                # BeaverTails-style: harmful prompt + harmful answer.
                prompt = row.get("prompt") or row.get("instruction") or ""
                answer = row.get("response") or row.get("answer") or row.get("output") or ""
                texts.append((prompt + " " + answer).strip() or prompt)
    if harmful_prompts:
        texts.extend(str(t) for t in harmful_prompts)
    if not texts:
        logger.info(
            "Antidote: no harmful realignment data supplied; using built-in "
            "default harmful prompt set (%d prompts).",
            len(_DEFAULT_HARMFUL_PROMPTS),
        )
        texts = list(_DEFAULT_HARMFUL_PROMPTS)
    texts = [t for t in texts if t][:max_samples]

    # 2. Tokenize. A tokenizer is required to turn harmful text into the
    #    integer ``input_ids`` a real HF LM expects -- this is what gives the
    #    WANDA score its safety signal.
    inputs: List[Any] = []
    if tokenizer is not None:
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=256)
            inputs.append({k: v.to(device) for k, v in enc.items()})
        return inputs

    # No tokenizer: try the model's own (some wrappers attach one).
    tok = getattr(model, "tokenizer", None)
    if tok is not None:
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=256)
            inputs.append({k: v.to(device) for k, v in enc.items()})
        return inputs

    logger.warning(
        "Antidote: no tokenizer available -- cannot encode the harmful "
        "realignment text into input_ids. Pass a `tokenizer=` kwarg for a "
        "real safety signal. Falling back to bare integer-id probes."
    )
    return []


@assert_mutates("apply_antidote")
def apply_antidote(
    model: nn.Module,
    prune_fraction: float = 0.05,
    target_modules: Optional[List[str]] = None,
    *,
    harmful_prompts: Optional[Sequence[str]] = None,
    dataset: Optional[Sequence[Any]] = None,
    tokenizer: Any = None,
    max_samples: int = 64,
    **extra: Any,
) -> nn.Module:
    """Apply Antidote WANDA pruning in-place to ``model``.

    One-shot prunes the **top-``prune_fraction``** WANDA-scored weights -- the
    weights most responsible for harmful generations -- to recover safe
    behaviour after harmful fine-tuning (arXiv:2408.09600).

    Parameters
    ----------
    model:
        The (harmfully fine-tuned) model to repair, pruned in place.
    prune_fraction:
        Fraction ``alpha`` of the highest-scoring weights to zero per target
        module (paper default 0.2; 0.05 used for GSM8K).
    target_modules:
        Substrings of Linear module names to prune (default
        ``["o_proj", "down_proj"]``, the WANDA-friendly projections).
    harmful_prompts:
        Optional list of harmful prompt strings used as the realignment
        calibration set. If omitted (and no ``dataset``), a small built-in
        harmful prompt set is used.
    dataset:
        Optional harmful realignment dataset -- a sequence of strings or of
        BeaverTails-style dicts (``prompt`` / ``response`` keys).
    tokenizer:
        Tokenizer used to encode the harmful text into ``input_ids``. Required
        for a real HF language model; if absent the model's own ``.tokenizer``
        is tried.
    max_samples:
        Cap on the number of realignment samples used for calibration.
    """
    if not isinstance(prune_fraction, (int, float)) or not 0.0 < prune_fraction < 1.0:
        raise ValueError(
            f"apply_antidote: prune_fraction must be in (0, 1), got {prune_fraction}"
        )

    targets = target_modules or ["o_proj", "down_proj"]

    try:
        device = next(model.parameters()).device
    except StopIteration as e:  # pragma: no cover - defensive
        raise ValueError("apply_antidote: model has no parameters.") from e

    # ── 1. Register hooks to accumulate input-activation L2 norms ────────────
    act_sq: dict = {}  # name -> sum of squared activations per input coordinate

    def _make_hook(name: str):
        def hook(module: nn.Module, inp: Any, output: Any) -> None:
            x = inp[0].detach()
            x = x.reshape(-1, x.size(-1)).float()
            # Accumulate sum of squares so the running value is a valid L2
            # norm across the *whole* realignment dataset (sqrt taken later).
            sq = (x * x).sum(dim=0)
            if name in act_sq:
                act_sq[name] = act_sq[name] + sq
            else:
                act_sq[name] = sq
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in targets):
            hooks.append(module.register_forward_hook(_make_hook(name)))

    # ── 2. Run the harmful realignment data through the model ────────────────
    realign_inputs = _build_realignment_inputs(
        model, harmful_prompts, dataset, tokenizer, device, max_samples
    )

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if realign_inputs:
                for batch in realign_inputs:
                    try:
                        model(**batch)
                    except Exception as e:  # pragma: no cover - defensive
                        logger.warning("Antidote: realignment forward failed: %s", e)
            else:
                # No tokenizer was available. Use integer token ids (not
                # float Gaussian noise) so a real LM embedding lookup works;
                # this is a degraded probe, not a true safety signal.
                try:
                    vocab = int(getattr(getattr(model, "config", None), "vocab_size", 32000))
                    probe = torch.randint(0, max(2, vocab), (2, 16), device=device)
                    model(input_ids=probe)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("Antidote: fallback probe forward failed: %s", e)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    if not act_sq:
        logger.warning(
            "Antidote: no activations were captured for target modules %s; "
            "no weights pruned.", targets
        )
        return model

    # ── 3. Compute WANDA score and prune the TOP-alpha (harmful) weights ─────
    pruned_count = 0
    total_count = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if name not in act_sq or not isinstance(module, nn.Linear):
                continue
            weight = module.weight.data
            # ||X||_2 per input coordinate over the whole realignment set.
            act_norm = torch.sqrt(act_sq[name].clamp_min(0)).to(weight.device)
            if act_norm.numel() != weight.size(1):
                logger.warning(
                    "Antidote: activation/weight shape mismatch for %s "
                    "(%d vs %d); skipping.",
                    name, act_norm.numel(), weight.size(1),
                )
                continue

            # WANDA score = |W| * ||X||  (broadcast act_norm over output rows).
            wanda_score = weight.abs().float() * act_norm.unsqueeze(0)

            # Standard WANDA structure: prune per output row. Within each row,
            # zero the top-`prune_fraction` HIGHEST-scoring weights -- the ones
            # the paper identifies as responsible for harmful generations.
            n_in = wanda_score.size(1)
            k = max(1, int(round(n_in * float(prune_fraction))))
            if k >= n_in:
                k = n_in - 1
            # Indices of the k largest scores per row -> these get zeroed.
            harmful_idx = torch.topk(wanda_score, k, dim=1, largest=True).indices
            prune_mask = torch.zeros_like(wanda_score, dtype=torch.bool)
            prune_mask.scatter_(1, harmful_idx, True)

            # w_tilde = (1 - m) * w  -- zero the harmful weights, keep the rest.
            weight.mul_((~prune_mask).to(weight.dtype))

            pruned_count += int(prune_mask.sum().item())
            total_count += wanda_score.numel()

    logger.info(
        "Antidote: pruned %d / %d weights (%.2f%%) across %d target modules.",
        pruned_count, total_count,
        100.0 * pruned_count / max(1, total_count),
        len(act_sq),
    )
    return model


__all__ = ["apply_antidote"]
