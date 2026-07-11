"""Counter-abliteration weight edit (SafeTune-original heuristic).

.. note::
   **This is a SafeTune-original training-free heuristic, not an
   implementation of any published method.** It is *not* the "DeepRefusal"
   paper. The historical module name is kept only for API stability.

Background
----------
Arditi et al. (2024, arXiv:2406.11717) showed that refusal in chat LLMs is
mediated by a single residual-stream direction; "abliteration" jailbreaks a
model by projecting that one direction out of every layer's output. A natural
training-free counter is to make the refusal signal *not* live in a single
direction, so that projecting out one direction leaves enough residual signal
that refusal still fires.

This module implements that idea directly in weight space:

  1. Given a refusal direction ``d`` (e.g. from
     :func:`safetune.steer.extract_refusal_direction`), for every targeted
     output-projection weight ``W`` (attention ``o_proj`` / MLP ``down_proj``)
     compute the component of ``W``'s output that lies along ``d``.
  2. Mirror that component onto a second, non-collinear direction ``d_perp``.
  3. After the edit, ``W`` writes the refusal-correlated signal into a 2-D
     subspace. An abliteration pass that only removes ``d`` leaves the
     ``d_perp`` component intact, so refusal partially survives.

Relation to published work
--------------------------
The published paper actually titled **DeepRefusal** -- Xie et al., "Beyond
Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically
Ablating Refusal Direction", Findings of EMNLP 2025
(https://aclanthology.org/2025.findings-emnlp.956/, arXiv:2509.15202, repo
https://github.com/YuanBoXie/DeepRefusal) -- is a **LoRA fine-tuning** method,
not a training-free weight edit. It registers forward hooks that, with a small
probability, *project the refusal direction out* of activations at random
layers and response-token positions, then fine-tunes the model so it learns to
re-emit refusal even from those ablated ("jailbroken") states. That method
needs a training loop, a dataset and gradient updates, so it cannot be a
Recover-pillar training-free ``apply_*`` patch and is **not** what this module
does. The two only share the high-level goal of hardening refusal against
direction-based attacks; the mechanisms are unrelated. No SafeTune module
currently implements the DeepRefusal paper.

This module is therefore judged on internal consistency alone.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ._invariant import assert_mutates
from safetune._refusal_helpers import _get_decoder_layers

logger = logging.getLogger(__name__)


def _spread_direction(d: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Return a unit vector orthogonal to ``d`` in the same ambient space.

    ``d`` is assumed already unit-normalised. The result is a deterministic
    (seeded) random unit vector with its ``d`` component removed.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    # Generate on CPU for deterministic RNG, then move to d's device so the
    # matmul below doesn't crash when `direction` lives on GPU.
    perp = torch.randn(d.shape[0], generator=g, dtype=torch.float32).to(d.device)
    d32 = d.float()
    perp = perp - (perp @ d32) * d32  # remove parallel component
    n = perp.norm()
    if n < 1e-12:
        # extremely unlikely; pick a basis vector and re-orthogonalize.
        perp = torch.zeros_like(d32)
        perp[0] = 1.0
        perp = perp - (perp @ d32) * d32
        n = perp.norm()
    return perp / n.clamp_min(1e-12)


@assert_mutates("apply_deeprefusal")
@torch.no_grad()
def apply_deeprefusal(
    model: nn.Module,
    direction: torch.Tensor,
    strength: float = 0.5,
    target_attn: bool = True,
    target_mlp: bool = True,
    seed: int = 0,
) -> nn.Module:
    """Spread the refusal signal across two non-collinear directions.

    SafeTune-original training-free counter-abliteration heuristic (see the
    module docstring -- this is **not** the DeepRefusal paper).

    For every targeted output-projection weight ``W`` in the model
    (attention ``o_proj`` and/or MLP ``down_proj``), we duplicate the
    component of ``W``'s output that lies along ``direction`` onto an
    orthogonal direction ``d_perp``. After the edit, ``W`` writes into a 2-D
    subspace whenever it previously wrote into the 1-D refusal subspace, so an
    abliteration pass that only removes ``direction`` leaves the ``d_perp``
    component intact.

    Args:
        model: HF causal LM.
        direction: refusal direction (1-D tensor of shape ``(hidden,)``).
        strength: how much of the parallel component to mirror onto
            ``d_perp``. ``1.0`` writes equal magnitude on both axes; ``0``
            is a no-op.
        target_attn: include attention output projections (``o_proj`` /
            ``out_proj`` / ``c_proj``).
        target_mlp: include MLP ``down_proj`` weights.
        seed: RNG seed for ``d_perp`` generation.

    Returns:
        The patched model (mutated in place).
    """
    layers = _get_decoder_layers(model)
    if not layers:
        raise RuntimeError(
            "apply_deeprefusal: cannot locate decoder layers. "
            "Expected model.model.language_model.layers (Gemma-3), "
            "model.model.layers (Llama/Mistral/Qwen/Gemma), or "
            "model.transformer.h (GPT-style)."
        )

    if direction.dim() != 1:
        raise ValueError(
            f"apply_deeprefusal: `direction` must be 1-D (hidden,), got "
            f"shape {tuple(direction.shape)}."
        )

    d = direction.detach().clone().float()
    norm = d.norm()
    if norm < 1e-12:
        raise ValueError("apply_deeprefusal: `direction` is a zero vector.")
    d = d / norm
    d_perp = _spread_direction(d, seed=seed)

    edited = 0
    skipped_dim = 0
    for layer in layers:
        for name, module in layer.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            is_attn = (
                name.endswith("o_proj")
                or name.endswith("out_proj")
                or name.endswith("c_proj")
            )
            is_mlp = name.endswith("down_proj")
            if not ((target_attn and is_attn) or (target_mlp and is_mlp)):
                continue
            # An output projection writes into the residual stream, so its
            # output dimension (rows of `weight`) must match `direction`.
            if module.weight.shape[0] != d.shape[0]:
                skipped_dim += 1
                continue
            dt = d.to(module.weight.dtype).to(module.weight.device)
            dp = d_perp.to(module.weight.dtype).to(module.weight.device)
            # Row-space coefficient: how each input feature contributes to the
            # output component along `d`.  parallel_coeff[j] = (dᵀ W)[j].
            parallel_coeff = (dt.unsqueeze(0) @ module.weight).squeeze(0)  # (in_dim,)
            # Mirror that component onto d_perp: W += strength · d_perp ⊗ (dᵀW).
            # This adds an output component along d_perp equal to `strength`
            # times the existing component along d, for every input.
            module.weight.data.add_(strength * torch.outer(dp, parallel_coeff))
            edited += 1

    if edited == 0:
        logger.warning(
            "apply_deeprefusal: no projection matrices matched (target_attn=%s, "
            "target_mlp=%s, %d skipped on dim mismatch). Model is unchanged.",
            target_attn, target_mlp, skipped_dim,
        )
    else:
        logger.info(
            "apply_deeprefusal: spread refusal signal across %d projection "
            "matrices (strength=%.2f, %d skipped on dim mismatch).",
            edited, strength, skipped_dim,
        )
    return model


__all__ = ["apply_deeprefusal"]
