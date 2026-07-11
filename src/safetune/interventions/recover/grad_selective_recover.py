"""Gradient-guided selective recovery (Yang et al., arXiv:2504.09757, Apr 2025).

Algorithm (paper Sec. 3 — gradient-guided rollback)
----------------------------------------------------
1. Forward + backward pass on a harmful calibration set to compute
   per-weight gradient saliency  S(w_ij) = |∂L_harmful / ∂w_ij|.
2. Select the top-``top_fraction`` weights by saliency — these are the
   weights most responsible for harmful generation.
3. For those coordinates, restore the value from the aligned reference
   checkpoint (rollback); unselected coordinates keep their drifted values.

This is a sparse, gradient-informed version of task arithmetic: only weights
the harm gradient identifies as safety-critical are rolled back to the aligned
model, leaving capability-encoding weights intact.

vLLM backend
------------
The gradient-saliency step requires PyTorch autograd (vLLM does not expose
gradients). When ``vllm_engine`` is supplied (a ``vllm.LLM`` instance) it is
used to *score* the most harmful calibration prompts before constructing the
gradient inputs, helping the saliency map focus on the highest-harm examples.
When ``None`` (default), all ``harmful_inputs`` are used directly as-is.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


def _saliency_map(
    model: nn.Module,
    inputs: List[torch.Tensor],
    target_names: List[str],
    max_samples: int,
) -> Dict[str, torch.Tensor]:
    """Accumulate |∂L_harmful/∂W| over harmful calibration inputs.

    Returns a dict mapping parameter name → saliency tensor (same shape as param).
    """
    saliency: Dict[str, torch.Tensor] = {}

    # Collect parameter references for the target modules.
    target_params: Dict[str, nn.Parameter] = {}
    for name, param in model.named_parameters():
        if any(t in name for t in target_names) and param.requires_grad:
            target_params[name] = param

    if not target_params:
        logger.warning(
            "grad_selective_recover: no trainable parameters found for target "
            "modules %s; saliency map will be empty.", target_names
        )
        return saliency

    was_training = model.training
    model.train()
    try:
        used = 0
        for ids in inputs[:max_samples]:
            if ids.numel() == 0:
                continue
            ids = ids.to(next(model.parameters()).device)
            # Teacher-force the full sequence as both input and label.
            # L_harmful = cross-entropy on the model's own harmful continuations.
            try:
                out = model(input_ids=ids, labels=ids)
                loss = out.loss if hasattr(out, "loss") else out[0]
                loss.backward()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("grad_selective_recover: forward/backward failed: %s", e)
                model.zero_grad(set_to_none=True)
                continue
            used += 1

        # Accumulate |grad| for each target param.
        with torch.no_grad():
            for name, param in target_params.items():
                if param.grad is not None:
                    g = param.grad.abs().float()
                    saliency[name] = saliency[name] + g if name in saliency else g.clone()
        model.zero_grad(set_to_none=True)

    finally:
        if not was_training:
            model.eval()

    logger.info(
        "grad_selective_recover: saliency map over %d harmful inputs, "
        "%d target params.", used if 'used' in dir() else 0, len(saliency)
    )
    return saliency


@assert_mutates("apply_grad_selective_recover")
def apply_grad_selective_recover(
    model: nn.Module,
    aligned: nn.Module,
    harmful_inputs: Sequence[torch.Tensor],
    *,
    top_fraction: float = 0.1,
    target_modules: Optional[List[str]] = None,
    max_samples: int = 32,
    vllm_engine: Any = None,
) -> nn.Module:
    """Gradient-guided selective weight rollback to the aligned checkpoint.

    Parameters
    ----------
    model:
        The drifted model to patch (mutated in-place).
    aligned:
        The safety reference checkpoint from which to restore the selected
        harmful-contributing weights.
    harmful_inputs:
        Sequence of ``input_ids`` tensors (shape ``(1, T)`` each) from
        tokenised harmful prompts used to drive the saliency computation.
    top_fraction:
        Fraction of weights per target parameter to restore from ``aligned``
        (selected by highest gradient saliency). Paper default: 0.1.
    target_modules:
        Substrings of parameter names to target (default: all Linear weight
        matrices that carry ``weight`` in their name).
    max_samples:
        Maximum number of harmful calibration samples used.
    vllm_engine:
        Optional ``vllm.LLM`` instance. When supplied, used to score and
        re-rank calibration prompts by harm probability before saliency
        computation (higher-harm examples first). Not required for correctness.

    Returns
    -------
    The mutated ``model``.
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")

    targets = target_modules or ["weight"]

    # Optional vLLM prompt ranking: use generation logprobs to put the most
    # harmful-activating inputs first so the saliency is concentrated on them.
    inputs = list(harmful_inputs) if harmful_inputs is not None else []
    if vllm_engine is not None:
        try:
            from vllm import SamplingParams  # type: ignore[import-not-found]
            sp = SamplingParams(max_tokens=1, logprobs=1)
            # Decode input_ids back to text for vLLM (vLLM takes strings).
            texts = []
            for ids in inputs[:max_samples]:
                tok = getattr(model, "tokenizer", None) or getattr(vllm_engine, "tokenizer", None)
                if tok is not None:
                    texts.append(tok.decode(ids[0].tolist(), skip_special_tokens=True))
            if texts:
                results = vllm_engine.generate(texts, sp)
                # Sort inputs by descending cumulative log-prob (more harmful first).
                scored = sorted(
                    zip(results, inputs[:max_samples]),
                    key=lambda x: (x[0].outputs[0].cumulative_logprob or 0.0),
                    reverse=True,
                )
                inputs = [s[1] for s in scored] + inputs[max_samples:]
                logger.info(
                    "grad_selective_recover: vLLM re-ranked %d prompts.", len(texts)
                )
        except Exception as e:  # pragma: no cover - optional dependency
            logger.debug(
                "grad_selective_recover: vLLM ranking skipped (%s); "
                "using original order.", e
            )

    # Compute gradient saliency map.
    saliency = _saliency_map(model, inputs, targets, max_samples)
    if not saliency:
        logger.warning(
            "grad_selective_recover: empty saliency map — no rollback performed."
        )
        return model

    aligned_sd = aligned.state_dict()
    model_sd = model.state_dict()

    restored_params = 0
    restored_coords = 0
    with torch.no_grad():
        for name, sal in saliency.items():
            if name not in aligned_sd or name not in model_sd:
                continue
            param_data = model_sd[name]
            aligned_param = aligned_sd[name].to(param_data.device, dtype=torch.float32)
            sal_f = sal.to(param_data.device).float()

            n = sal_f.numel()
            k = max(1, int(round(n * top_fraction)))
            if k >= n:
                k = n

            # Flatten, topk, build restore mask.
            flat_sal = sal_f.flatten()
            top_idx = torch.topk(flat_sal, k, largest=True).indices
            mask = torch.zeros(n, dtype=torch.bool, device=param_data.device)
            mask.scatter_(0, top_idx, True)
            mask = mask.view(param_data.shape)

            # Restore selected coordinates from aligned; keep rest from drifted.
            param_f = param_data.float()
            new_val = torch.where(mask, aligned_param, param_f)
            param_data.copy_(new_val.to(param_data.dtype))

            restored_params += 1
            restored_coords += int(mask.sum().item())

    model.load_state_dict(model_sd, strict=False)
    logger.info(
        "grad_selective_recover: restored %d coords across %d params "
        "(top_fraction=%.3f).",
        restored_coords, restored_params, top_fraction,
    )
    return model


__all__ = ["apply_grad_selective_recover"]
