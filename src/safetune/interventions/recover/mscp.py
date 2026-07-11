"""MSCP: Multi-Level Safety Continual Projection (training-free).

Reference paper: "Fine-Grained Safety Neurons with Training-Free Continual
Projection to Reduce LLM Fine Tuning Risks" / "Multi-Level Safety Continual
Projection for Fine-Tuned Large Language Models without Retraining"
(arXiv:2508.09190). No official code release is published by the authors; this
module reimplements the projection mechanism from the paper's equations.

The full MSCP pipeline has three parts:

  1. **Layer localisation** — identify safety-critical layers (roughly the
     layers around one-third of the model depth) where benign-vs-harmful hidden
     states diverge most.
  2. **Fine-grained neuron localisation** — within those layers, score neurons
     by activation importance and build a binary mask ``Mask_l`` selecting the
     top-q% neurons on harmful data while *excluding* the top-p% on benign data
     (paper Eq. 4-5).
  3. **Sparse safety projection** — build the safety projection matrix
     ``W_safe = (W_align - W_base)(W_align - W_base)^T / Dim`` (Eq. 6) from a
     publicly available aligned/base model pair, and project only the masked
     safety neurons: ``Proj_safe(W) = Mask_l . W_safe . W`` (Eq. 7).

Parts 1-2 require benign/harmful activation contrasts (data + extra reference
models) and are *out of scope* for a training-free, data-free Recover patch:
this module therefore consumes their two artefacts as caller-supplied inputs:

  * an aligned/base model pair (``aligned_state``/``base_state`` or their
    ``*_param_path`` variants) from which ``W_safe`` is computed per Eq. 6, and
  * an optional per-parameter neuron mask ``neuron_mask`` (the ``Mask_l`` of
    Eq. 4); without it the projection is applied to every output neuron of each
    matched 2-D weight.

When that artefact pair is supplied, ``apply_mscp`` runs the **faithful MSCP
projection** (Eq. 6-7). When it is not, the call falls back to the legacy
:class:`MSCPProjectionPatch` primitive (subtract / orthogonalise against a
single user-supplied direction vector) so existing callers and the dict-mode
tests are unaffected.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


def _load_state_dict(path: str) -> Dict[str, Any]:
    """Load a checkpoint as a flat ``param-name -> tensor`` dict.

    Supports a raw state dict, a ``{"model": state_dict}`` wrapper, and
    ``.safetensors`` files.
    """
    import torch as _torch

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file  # type: ignore[import-not-found]

        return dict(load_file(path, device="cpu"))
    raw = _torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict):
        return raw.get("model", raw)
    return raw


@assert_mutates("apply_mscp")
def apply_mscp(
    model: nn.Module,
    direction: Union[List[float], Dict[str, List[float]], None] = None,
    coefficient: float = 1.0,
    mode: str = "subtract",
    aligned_state: Optional[Dict[str, Any]] = None,
    aligned_param_path: Optional[str] = None,
    base_state: Optional[Dict[str, Any]] = None,
    base_param_path: Optional[str] = None,
    neuron_mask: Optional[Dict[str, Any]] = None,
    neuron_mask_path: Optional[str] = None,
    **extra: Any,
) -> nn.Module:
    """Apply Multi-Level Safety Continual Projection to ``model`` in place.

    Faithful to MSCP (arXiv:2508.09190). Two paths:

    **Faithful MSCP path** — triggered when *both* an aligned and a base model
    are supplied. For each matched 2-D weight ``W`` (shape ``(out, in)``):

      1. ``D = W_align - W_base`` (the alignment delta);
      2. ``W_safe = (D @ D^T) / Dim(D)`` — the safety projection matrix of
         Eq. 6, an ``(out, out)`` symmetric PSD matrix;
      3. ``W_proj = W_safe @ W`` — the projection of Eq. 7;
      4. at the safety-neuron rows selected by ``Mask_l`` (``neuron_mask``),
         blend toward the projection by ``coefficient``:
         ``W_row <- (1 - coefficient) * W_row + coefficient * W_proj_row``.
         Rows outside the mask are left untouched. ``coefficient = 1.0``
         reproduces the paper's exact replacement projection
         ``W <- Mask_l . W_safe . W``.

    .. warning::
        ``coefficient = 1.0`` (the default, matching the paper's Eq. 7) applies
        the full safety projection to every 2-D weight. On models larger than
        ~1B parameters this tends to cause significant capability collapse
        because the projection over-constrains all output neurons toward the
        alignment subspace. If you observe degraded utility, lower
        ``coefficient`` to the range ``0.05–0.2``. The SafeTune audit used
        ``coefficient = 0.05`` for this reason.

    **Legacy path** — when no aligned/base pair is given, delegates to
    :class:`safetune.core.patches.mscp_projection.MSCPProjectionPatch`, which
    subtracts / orthogonalises each parameter against a single user-supplied
    ``direction`` vector. This is a generic projection primitive, *not* the
    MSCP algorithm; it is kept only for backward compatibility.

    Args:
        model: the fine-tuned ``nn.Module`` to realign, mutated in place.
        direction: legacy-path safety direction — a flat list applied to every
            parameter, or a ``{param_name: list}`` dict. Ignored on the
            faithful path.
        coefficient: projection strength. On the faithful path it interpolates
            between the original weight (``0.0``) and the full safety
            projection (``1.0``, the paper's behaviour). On the legacy path it
            is the subtract/orthogonalise coefficient.
        mode: legacy-path mode, ``"subtract"`` or ``"orthogonal"``.
        aligned_state: in-memory state dict of the safety-aligned reference
            model (the publicly available *-Instruct* checkpoint in the paper).
        aligned_param_path: path to the aligned checkpoint (alternative to
            ``aligned_state``; ``.safetensors`` supported).
        base_state: in-memory state dict of the unaligned base model.
        base_param_path: path to the base checkpoint.
        neuron_mask: per-parameter binary mask ``Mask_l`` (Eq. 4). For a weight
            of shape ``(out, in)`` this may be a length-``out`` 1-D tensor (a
            per-output-neuron row mask) or a full ``(out, in)`` tensor. ``True``
            marks the safety-critical neurons to project. Absent -> every
            output neuron of each matched weight is projected.
        neuron_mask_path: path to a saved ``neuron_mask`` dict (``torch.save``).
        **extra: forwarded to ``MSCPProjectionPatch`` on the legacy path.

    Returns:
        The mutated ``model``.
    """
    # Resolve the aligned/base reference pair (the W_safe artefact, Eq. 6).
    resolved_aligned: Optional[Dict[str, Any]] = None
    if aligned_state is not None:
        resolved_aligned = dict(aligned_state)
    elif aligned_param_path is not None:
        try:
            resolved_aligned = _load_state_dict(aligned_param_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "apply_mscp: could not load aligned model from %s — %s; "
                "falling back to the legacy projection primitive.",
                aligned_param_path,
                exc,
            )
            resolved_aligned = None

    resolved_base: Optional[Dict[str, Any]] = None
    if base_state is not None:
        resolved_base = dict(base_state)
    elif base_param_path is not None:
        try:
            resolved_base = _load_state_dict(base_param_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "apply_mscp: could not load base model from %s — %s; "
                "falling back to the legacy projection primitive.",
                base_param_path,
                exc,
            )
            resolved_base = None

    # Resolve the per-parameter neuron mask (the Mask_l artefact, Eq. 4).
    resolved_mask: Optional[Dict[str, Any]] = None
    if neuron_mask is not None:
        resolved_mask = dict(neuron_mask)
    elif neuron_mask_path is not None:
        import torch as _torch

        try:
            resolved_mask = dict(_torch.load(neuron_mask_path, map_location="cpu", weights_only=True))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "apply_mscp: could not load neuron_mask from %s — %s",
                neuron_mask_path,
                exc,
            )
            resolved_mask = None

    # Faithful MSCP path: requires both reference models (Eq. 6 needs both).
    if resolved_aligned is not None and resolved_base is not None:
        _mscp_safety_projection(
            model,
            aligned=resolved_aligned,
            base=resolved_base,
            neuron_mask=resolved_mask,
            coefficient=float(coefficient),
        )
        return model

    if resolved_aligned is not None or resolved_base is not None:
        logger.warning(
            "apply_mscp: MSCP needs *both* an aligned and a base model to "
            "build the safety projection matrix W_safe (Eq. 6); only one was "
            "supplied. Falling back to the legacy projection primitive."
        )

    # Legacy path: generic subtract/orthogonalise against a direction vector.
    try:
        from safetune.core.patches.mscp_projection import (
            MSCPProjectionPatch,
        )
    except ImportError as e:  # pragma: no cover - defensive
        raise ImportError(
            f"apply_mscp needs safetune.core.patches.mscp_projection: {e}"
        ) from e

    params: Dict[str, Any] = {
        "direction": direction if direction is not None else [],
        "coefficient": coefficient,
        "mode": mode,
    }
    params.update(extra)

    patch = MSCPProjectionPatch(**params)
    patch.apply_to_model(model)
    return model


def _mscp_safety_projection(
    model: nn.Module,
    aligned: Dict[str, Any],
    base: Dict[str, Any],
    neuron_mask: Optional[Dict[str, Any]],
    coefficient: float,
) -> None:
    """Faithful MSCP sparse safety projection — paper Eq. 6-7.

    For every named 2-D weight ``W`` (shape ``(out, in)``) that has both an
    aligned and a base counterpart:

        D       = W_align - W_base                      (alignment delta)
        W_safe  = (D @ D^T) / numel(D)                  (Eq. 6, (out, out) PSD)
        W_proj  = W_safe @ W                            (Eq. 7 projection)
        W_row  <- (1 - c) * W_row + c * W_proj_row      at masked neuron rows

    ``W_safe`` is the symmetric positive-semidefinite Gram matrix of the
    alignment delta: projecting ``W`` through it re-weights each output
    neuron's row toward the directions in which the aligned and base models
    most disagree (the "safety directions"). Rows not selected by ``Mask_l``
    are left untouched, giving the paper's *sparse* edit. ``coefficient = 1.0``
    reproduces the exact paper projection ``W <- Mask_l . W_safe . W``.
    """
    import torch as _torch

    projected = 0
    skipped_shape = 0
    total = 0

    with _torch.no_grad():
        for name, param in model.named_parameters():
            if name not in aligned or name not in base:
                continue
            # Eq. 6 builds W_safe = D @ D^T, which is only defined for a 2-D
            # weight matrix. 1-D params (biases, norms) have no neuron-level
            # projection in MSCP and are left untouched.
            if param.dim() != 2:
                continue
            total += 1

            w_align = aligned[name].to(device=param.device, dtype=_torch.float32)
            w_base = base[name].to(device=param.device, dtype=_torch.float32)
            if w_align.shape != param.shape or w_base.shape != param.shape:
                logger.warning(
                    "apply_mscp: aligned/base shape mismatch for %s "
                    "(%s / %s vs %s); skipping.",
                    name,
                    tuple(w_align.shape),
                    tuple(w_base.shape),
                    tuple(param.shape),
                )
                skipped_shape += 1
                continue

            out_dim = param.shape[0]

            # Eq. 6+7 — W_proj = W_safe @ W with W_safe = (D @ Dᵀ) / Dim(D).
            # Computed associatively as D @ (Dᵀ @ W) / Dim(D) so the (out, out)
            # Gram matrix W_safe is never materialized: for the vocab matrices
            # (lm_head / embed_tokens, out = vocab ≈ 128k) a dense (out, out)
            # fp32 tensor is ~65 GB and OOMs. The intermediate Dᵀ @ W is only
            # (in, in); the projection is numerically identical.
            delta = w_align - w_base
            denom = float(delta.numel()) or 1.0
            w_fp = param.data.to(_torch.float32)
            w_proj = (delta @ (delta.transpose(0, 1) @ w_fp)) / denom  # (out, in)

            # Mask_l (Eq. 4): a per-output-neuron row mask. A full (out, in)
            # mask is collapsed to a row mask (a neuron is selected if any of
            # its weights are). Absent -> all output neurons projected.
            if neuron_mask is not None and name in neuron_mask:
                mask = neuron_mask[name].to(device=param.device).bool()
                if mask.dim() == 2 and mask.shape == param.shape:
                    row_mask = mask.any(dim=1)
                elif mask.dim() == 1 and mask.shape[0] == out_dim:
                    row_mask = mask
                else:
                    logger.warning(
                        "apply_mscp: neuron_mask shape %s for %s does not "
                        "match an (out,in) or (out,) mask; projecting all "
                        "output neurons.",
                        tuple(mask.shape),
                        name,
                    )
                    row_mask = _torch.ones(out_dim, dtype=_torch.bool,
                                           device=param.device)
            else:
                row_mask = _torch.ones(out_dim, dtype=_torch.bool,
                                       device=param.device)

            if not bool(row_mask.any()):
                continue

            # Sparse blend toward the projection at the masked neuron rows.
            new_w = w_fp.clone()
            new_w[row_mask] = (
                (1.0 - coefficient) * w_fp[row_mask]
                + coefficient * w_proj[row_mask]
            )
            param.data.copy_(new_w.to(param.dtype))
            projected += 1

    logger.info(
        "apply_mscp: faithful MSCP projection — %d/%d 2-D weights projected "
        "(%d skipped on shape mismatch, neuron_mask=%s, coefficient=%.3f).",
        projected,
        total,
        skipped_shape,
        neuron_mask is not None,
        coefficient,
    )


__all__ = ["apply_mscp"]
