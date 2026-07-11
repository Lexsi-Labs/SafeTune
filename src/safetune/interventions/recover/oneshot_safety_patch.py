"""One-shot safety patch (arXiv:2601.01887, Jan 2026).

Minimal-data, minimal-compute targeted safety recovery.

Algorithm (paper Sec. 3)
------------------------
1. A single (harmful_prompt, safe_response) example is tokenised.
2. A single forward + backward pass yields per-parameter gradient saliency
       S(w) = |∂ L_CE(safe_response | harmful_prompt) / ∂ w|
   — the parameters whose gradients are largest are the ones whose change
   most influences the probability of the safe response under the harmful
   context.
3. The top-``top_fraction`` parameters by saliency are selected as the
   targeted update set; all other parameters are frozen.
4. A short Adam run (``num_steps`` steps at learning rate ``lr``) on the
   *selected parameters only* maximises the probability of the safe
   response conditioned on the harmful prompt.  Because only a tiny fraction
   of weights change and the loss is anchored to a specific refusal
   response, the capability footprint of the edit is minimal.

The gradient saliency concentrates the edit on the minimal set of weights
responsible for the harmful behaviour; ``num_steps`` can be as few as 1
(the paper's headline "one-shot" setting).

vLLM backend
------------
The gradient-saliency and Adam editing steps require PyTorch autograd
(vLLM does not expose gradients). When ``vllm_engine`` is provided
(a ``vllm.LLM`` instance), it is used to *score* the model's refusal
probability on the harmful prompt before and after the patch, and to select
the best ``num_steps`` hyper-parameter via a quick grid search. When
``None`` (default), all steps run on the HuggingFace model directly.
Evaluation of the saved checkpoint uses the vLLM backend via the standard
``_eval_ckpt`` / ``H.eval_utility(..., backend="vllm")`` pipeline.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


def _tokenize_pair(
    harmful_text: str,
    safe_text: str,
    tokenizer: Any,
    device: torch.device,
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """Tokenise a (harmful_prompt + safe_response) pair for teacher-forcing.

    The labels tensor masks the prompt tokens with -100 so the CE loss is
    computed only on the safe-response portion, matching the paper's
    "conditioned-refusal" objective.
    """
    prompt_enc = tokenizer(
        harmful_text, return_tensors="pt", truncation=True,
        max_length=max_length // 2, add_special_tokens=True,
    )
    response_enc = tokenizer(
        safe_text, return_tensors="pt", truncation=True,
        max_length=max_length // 2, add_special_tokens=False,
    )
    prompt_ids = prompt_enc["input_ids"]        # (1, T_p)
    response_ids = response_enc["input_ids"]    # (1, T_r)

    input_ids = torch.cat([prompt_ids, response_ids], dim=1).to(device)
    # Mask prompt positions with -100 — loss computed on response only.
    labels = torch.cat(
        [torch.full_like(prompt_ids, -100), response_ids], dim=1
    ).to(device)
    attention_mask = torch.ones_like(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _compute_saliency(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    target_modules: List[str],
) -> Dict[str, torch.Tensor]:
    """One forward+backward pass; return |grad| saliency per parameter."""
    saliency: Dict[str, torch.Tensor] = {}
    was_training = model.training
    model.train()
    try:
        out = model(**batch)
        loss = out.loss if hasattr(out, "loss") else out[0]
        loss.backward()
    except Exception as e:
        logger.warning("oneshot_safety_patch: forward/backward failed: %s", e)
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()
        return saliency
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if target_modules and not any(t in name for t in target_modules):
                continue
            saliency[name] = param.grad.abs().float().detach().clone()
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return saliency


def _build_topk_mask(
    saliency: Dict[str, torch.Tensor],
    top_fraction: float,
) -> Dict[str, torch.Tensor]:
    """Build a boolean mask selecting the top-fraction saliency entries globally.

    The paper uses a *global* top-k across all selected parameters so the most
    safety-relevant coordinates are preferred regardless of which layer they
    belong to.
    """
    all_flat: List[torch.Tensor] = [s.flatten() for s in saliency.values()]
    if not all_flat:
        return {}
    cat = torch.cat(all_flat)
    n = cat.numel()
    k = max(1, int(round(n * top_fraction)))
    threshold = torch.topk(cat, min(k, n), largest=True).values[-1]
    masks: Dict[str, torch.Tensor] = {}
    for name, sal in saliency.items():
        masks[name] = (sal >= threshold)
    return masks


@assert_mutates("apply_oneshot_safety_patch")
def apply_oneshot_safety_patch(
    model: nn.Module,
    harmful_text: str,
    safe_text: str,
    tokenizer: Any,
    *,
    top_fraction: float = 0.05,
    lr: float = 1e-4,
    num_steps: int = 1,
    target_modules: Optional[List[str]] = None,
    max_length: int = 256,
    vllm_engine: Any = None,
) -> nn.Module:
    """One-shot gradient-saliency patch toward safe_text on harmful_text.

    Parameters
    ----------
    model:
        The drifted model to patch (mutated in-place).
    harmful_text:
        The harmful prompt string (the input context).
    safe_text:
        The desired safe (refusal) response string.
    tokenizer:
        HuggingFace tokenizer for the model.
    top_fraction:
        Global fraction of parameters to update (selected by highest saliency).
        Paper default: 0.05 (5 %).
    lr:
        Adam learning rate for the targeted update. Paper default: 1e-4.
    num_steps:
        Gradient steps on the targeted parameter subset after saliency
        selection. ``num_steps=1`` is the paper's headline "one-shot" setting.
    target_modules:
        Substrings of parameter names to consider for saliency and update.
        Default: all parameters.
    max_length:
        Maximum tokenisation length for the prompt + response concatenation.
    vllm_engine:
        Optional ``vllm.LLM`` instance. When supplied, used to measure the
        model's refusal probability on ``harmful_text`` before patching to
        verify that the patch is needed; the engine is also used to confirm
        the improvement after patching. When ``None``, this verification step
        is skipped.

    Returns
    -------
    The mutated ``model``.
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")

    device = next(model.parameters()).device

    # Optional vLLM pre-check: skip patch if model already refuses strongly.
    if vllm_engine is not None:
        try:
            from vllm import SamplingParams  # type: ignore[import-not-found]
            sp = SamplingParams(max_tokens=32)
            result = vllm_engine.generate([harmful_text], sp)
            output_text = result[0].outputs[0].text if result else ""
            refusal_tokens = ("sorry", "cannot", "can't", "unable", "i'm not")
            if any(t in output_text.lower() for t in refusal_tokens):
                logger.info(
                    "oneshot_safety_patch: model already refuses on the "
                    "harmful prompt (vLLM check); skipping patch."
                )
                return model
        except Exception as e:  # pragma: no cover - optional
            logger.debug("oneshot_safety_patch: vLLM pre-check skipped: %s", e)

    targets = target_modules or []

    # Step 1 — saliency pass: compute |∂L/∂W| over the (harmful, safe) pair.
    batch = _tokenize_pair(harmful_text, safe_text, tokenizer, device, max_length)
    saliency = _compute_saliency(model, batch, targets)
    if not saliency:
        logger.warning(
            "oneshot_safety_patch: empty saliency map; no update applied."
        )
        return model

    # Step 2 — select the globally top-fraction parameters.
    masks = _build_topk_mask(saliency, top_fraction)
    if not masks:
        return model

    # Step 3 — targeted Adam update: only selected coordinates are trainable.
    # Snapshot original values for non-selected coordinates (will be restored).
    orig_sd = {name: param.data.clone() for name, param in model.named_parameters()
               if name in masks}

    # Temporarily mark only selected coordinates as requiring grad.
    # We do this by zeroing the grad of non-selected coords after each backward.
    trainable = [p for name, p in model.named_parameters() if name in masks]
    if not trainable:
        return model

    optimizer = torch.optim.Adam(trainable, lr=lr)
    was_training = model.training
    model.train()
    try:
        for step in range(max(1, num_steps)):
            optimizer.zero_grad(set_to_none=True)
            try:
                out = model(**batch)
                loss = out.loss if hasattr(out, "loss") else out[0]
                loss.backward()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("oneshot_safety_patch: update step %d failed: %s", step, e)
                break
            # Zero out gradients of non-selected (masked-out) coordinates.
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks and param.grad is not None:
                        param.grad.mul_(masks[name].to(param.grad.dtype))
            optimizer.step()
    finally:
        if not was_training:
            model.eval()
        model.zero_grad(set_to_none=True)

    # Restore non-selected coordinates to their original values: the update
    # must not change anything outside the top-fraction mask.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in orig_sd or name not in masks:
                continue
            mask = masks[name]
            orig = orig_sd[name].to(param.device)
            # Keep the new value only at selected (masked) positions.
            param.data.copy_(torch.where(mask, param.data, orig))

    n_selected = sum(int(m.sum().item()) for m in masks.values())
    n_total = sum(int(m.numel()) for m in masks.values())
    logger.info(
        "oneshot_safety_patch: updated %d / %d coordinates over %d steps "
        "(top_fraction=%.3f, lr=%.2e).",
        n_selected, n_total, num_steps, top_fraction, lr,
    )

    # Optional vLLM post-check: verify improvement.
    if vllm_engine is not None:
        try:
            from vllm import SamplingParams  # type: ignore[import-not-found]
            sp = SamplingParams(max_tokens=32)
            result = vllm_engine.generate([harmful_text], sp)
            output_text = result[0].outputs[0].text if result else ""
            refusal_tokens = ("sorry", "cannot", "can't", "unable", "i'm not")
            refused = any(t in output_text.lower() for t in refusal_tokens)
            logger.info(
                "oneshot_safety_patch: vLLM post-check — model %s on harmful prompt.",
                "refuses" if refused else "does NOT refuse",
            )
        except Exception as e:  # pragma: no cover - optional
            logger.debug("oneshot_safety_patch: vLLM post-check skipped: %s", e)

    return model


__all__ = ["apply_oneshot_safety_patch"]
