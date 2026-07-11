from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence

import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


def _matched_param_keys(
    state_dict_keys: Iterable[str],
    layer_subset: Optional[Sequence[int]],
    target_modules: Optional[Sequence[str]],
) -> List[str]:
    """Return state-dict keys that match the (layer_subset, target_modules) filter."""
    keys = list(state_dict_keys)
    if layer_subset is None and not target_modules:
        return [k for k in keys if k.endswith(".weight")]

    # FIX: Explicitly distinguish between None (no filter) and [] (match nothing)
    layer_set = set(int(i) for i in layer_subset) if layer_subset is not None else None
    
    out: List[str] = []
    for k in keys:
        if not k.endswith(".weight"):
            continue
        if layer_set is not None:
            parts = k.split(".")
            try:
                # Assumes the first integer in the key is the layer index
                layer_idx = next(int(p) for p in parts if p.isdigit())
            except StopIteration:
                continue
            if layer_idx not in layer_set:
                continue
        if target_modules:
            if not any(tm in k for tm in target_modules):
                continue
        out.append(k)
    return out


@assert_mutates("apply_ctheta")
def apply_ctheta(
    target: nn.Module,
    positive: nn.Module,
    negative: nn.Module,
    circuit_info,
    strength: float = 1.0,
    layer_subset: Optional[Sequence[int]] = None,
    target_modules: Optional[Sequence[str]] = None,
) -> nn.Module:
    """Apply circuit-guided weight steering in-place."""
    sd_t = target.state_dict()
    sd_p = positive.state_dict()
    sd_n = negative.state_dict()

    suggestions = getattr(circuit_info, "layer_suggestions", None)
    if layer_subset is None and suggestions is not None:
        layer_subset = getattr(suggestions, "layer_subset", None)
    if target_modules is None and suggestions is not None:
        target_modules = getattr(suggestions, "target_modules", None) or None

    matched = _matched_param_keys(sd_t.keys(), layer_subset, target_modules)
    if not matched:
        raise ValueError(
            "apply_ctheta: no parameters matched the circuit mask "
            f"(layer_subset={layer_subset}, target_modules={target_modules})"
        )

    logger.info("apply_ctheta: steering %d parameters @ strength %.3f",
                len(matched), strength)

    n_applied = 0
    for k in matched:
        if k not in sd_p or k not in sd_n:
            continue
        tp = sd_t[k]
        if not (tp.shape == sd_p[k].shape == sd_n[k].shape):
            continue
        delta = sd_p[k].to(tp.device, dtype=tp.dtype) - sd_n[k].to(tp.device, dtype=tp.dtype)
        sd_t[k] = tp + strength * delta
        n_applied += 1

    target.load_state_dict(sd_t, strict=False)
    logger.info("apply_ctheta: applied steering to %d / %d matched keys", n_applied, len(matched))
    return target


@assert_mutates("apply_ctheta_from_state_dicts")
def apply_ctheta_from_state_dicts(
    target: nn.Module,
    positive_sd: dict,
    negative_sd: dict,
    circuit_info,
    strength: float = 1.0,
    layer_subset: Optional[Sequence[int]] = None,
    target_modules: Optional[Sequence[str]] = None,
) -> nn.Module:
    """In-memory variant: skips re-loading the positive / negative models."""
    sd_t = target.state_dict()
    suggestions = getattr(circuit_info, "layer_suggestions", None)
    if layer_subset is None and suggestions is not None:
        layer_subset = getattr(suggestions, "layer_subset", None)
    if target_modules is None and suggestions is not None:
        target_modules = getattr(suggestions, "target_modules", None) or None

    matched = _matched_param_keys(sd_t.keys(), layer_subset, target_modules)
    if not matched:
        raise ValueError("apply_ctheta_from_state_dicts: no matched parameters")

    n_applied = 0
    for k in matched:
        if k not in positive_sd or k not in negative_sd:
            continue
        tp = sd_t[k]
        if not (tp.shape == positive_sd[k].shape == negative_sd[k].shape):
            continue
        delta = positive_sd[k].to(tp.device, dtype=tp.dtype) - negative_sd[k].to(tp.device, dtype=tp.dtype)
        sd_t[k] = tp + strength * delta
        n_applied += 1

    target.load_state_dict(sd_t, strict=False)
    return target


def sweep_ctheta_strength(
    target: nn.Module,
    positive: nn.Module,
    negative: nn.Module,
    circuit_info,
    strengths: Sequence[float],
    eval_fn,
    layer_subset: Optional[Sequence[int]] = None,
    target_modules: Optional[Sequence[str]] = None,
    higher_is_better: bool = False,
) -> List[dict]:
    """Sweep the steering coefficient one strength at a time."""
    sd_p = positive.state_dict()
    sd_n = negative.state_dict()

    suggestions = getattr(circuit_info, "layer_suggestions", None)
    if layer_subset is None and suggestions is not None:
        layer_subset = getattr(suggestions, "layer_subset", None)
    if target_modules is None and suggestions is not None:
        target_modules = getattr(suggestions, "target_modules", None) or None

    sd_t = target.state_dict()
    matched = _matched_param_keys(sd_t.keys(), layer_subset, target_modules)
    if not matched:
        raise ValueError("sweep_ctheta_strength: no parameters matched the circuit mask")

    # FIX 1: Safely compute delta on CPU to avoid device mismatch crashes and save GPU VRAM.
    delta_sd = {}
    for k in matched:
        if k in sd_p and k in sd_n and sd_p[k].shape == sd_n[k].shape:
            delta_sd[k] = sd_p[k].cpu() - sd_n[k].cpu()

    # FIX 2: Cache the original weights of the matched circuit to guarantee 
    # a bit-identical revert and eliminate floating-point drift.
    # Since C-ΔΘ masks are inherently sparse, this memory footprint is negligible.
    original_sd = {k: sd_t[k].clone() for k in matched}

    rows: List[dict] = []
    prev_strength = 0.0
    
    try:
        for k_val in strengths:
            step = k_val - prev_strength
            
            # Apply incremental delta to avoid an O(strengths) re-copy.
            for key, d in delta_sd.items():
                tp = sd_t[key]
                sd_t[key] = tp + step * d.to(tp.device, dtype=tp.dtype)
            
            target.load_state_dict(sd_t, strict=False)
            
            # FIX 3: Update prev_strength BEFORE eval_fn. If eval_fn throws an error, 
            # the mathematical state of the model is accurately tracked for the finally block.
            prev_strength = k_val
            
            score = float(eval_fn(target))
            rows.append({"strength": float(k_val), "score": score, "best": False})
            logger.info("sweep_ctheta_strength: k=%.3f score=%.4f", k_val, score)
            
    finally:
        # FIX 4: Restore directly from the cloned original weights. 
        # This guarantees 100% bit-identical recovery regardless of exceptions or FP drift.
        target.load_state_dict(original_sd, strict=False)

    if rows:
        cmp_key = (lambda r: -r["score"]) if higher_is_better else (lambda r: r["score"])
        best = min(rows, key=cmp_key)
        best["best"] = True
    return rows


__all__ = [
    "apply_ctheta",
    "apply_ctheta_from_state_dicts",
    "sweep_ctheta_strength",
]