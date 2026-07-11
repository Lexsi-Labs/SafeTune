"""
CRISP: Persistent Concept Unlearning via Sparse Autoencoders
(Gao et al., arXiv:2508.13650, 2025).

"CRISP: Persistent Concept Unlearning via Sparse Autoencoders" — 2025.

Standard SAE-based unlearning ablates concept features *at inference time*:
the SAE must be present at every forward pass, making deployment expensive.
CRISP instead **fine-tunes the model weights** so that the target SAE features
are suppressed at the source, making the unlearning permanent and runtime-free.

⚠️ FIDELITY: this is a **simplified SafeTune variant**, NOT a faithful
reproduction of arXiv:2508.13650 §3. It captures the core idea (weight-space
suppression of SAE concept features) but differs from the paper in three ways:
(1) the suppression loss is a squared-L2 on concept features rather than the
paper's raw signed activation term plus the ``λ·c_t`` mean-activation
regulariser; (2) the retain term is standard cross-entropy rather than the
paper's ``‖h_M − h_M0‖²`` representation-matching (+ coherence) loss; (3) it
operates on a single layer rather than a mean over a layer subset. Treat as
"inspired by CRISP," not the published algorithm.

Algorithm (simplified from arXiv:2508.13650 §3):

  Let ``h_ℓ(θ; x)`` be the residual-stream activation at layer ``ℓ`` for
  input ``x`` under model parameters ``θ``.  Let ``f = SAE.encode(h_ℓ)`` be
  the SAE feature vector.  ``f_concept`` is the sub-vector indexed by
  ``concept_features``.

  CRISP minimises::

      L_CRISP(θ) = L_retain(θ; retain_data)
                 + γ · ‖f_concept(h_ℓ(θ; forget_data))‖₂²

  * ``L_retain`` is the standard causal-LM cross-entropy loss on the retain
    set, keeping utility intact.
  * The suppression term ``γ · ‖f_concept(h)‖₂²`` drives the activations of
    the target SAE features to zero on the forget set, baking the suppression
    into ``θ`` rather than requiring a runtime SAE ablation hook.

  The SAE is frozen throughout.  Only model parameters receive gradients.
  The hook that extracts ``h_ℓ`` is installed once per step via
  ``register_forward_hook`` and removed immediately after, matching the
  authors' approach of coupling the SAE at a single intermediate layer.

Implementation notes:

  * ``hook_layer=-1`` selects the last decoder block.  Any valid layer index
    is accepted.
  * The SAE must expose a ``.encode(h) -> features`` method where ``h`` has
    shape ``(batch, seq, hidden)`` or ``(batch, hidden)``; the method returns
    feature activations of shape ``(batch, [seq,] n_features)``.
  * Batches in ``forget_dataset`` and ``retain_dataset`` should be dicts of
    tensors (e.g. ``{"input_ids": ..., "attention_mask": ..., "labels": ...}``).
    If ``"labels"`` is absent in a retain batch the cross-entropy loss is
    skipped for that step and only the suppression term is optimised.
  * All trainable model parameters are updated (no sub-selection), matching
    the full-weight fine-tuning described in the paper.

Reference: L. Gao, T. Hua, A. Khodabandeh, et al., "CRISP: Persistent
Concept Unlearning via Sparse Autoencoders." arXiv:2508.13650 (2025).
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from safetune._refusal_helpers import _get_decoder_layers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_hook_layer(model: nn.Module, hook_layer: int) -> nn.Module:
    """Return the decoder block at ``hook_layer``.

    ``hook_layer=-1`` selects the last block.
    """
    layers = _get_decoder_layers(model)
    if not layers:
        raise ValueError(
            "crisp_unlearn: could not locate decoder layers. "
            "Expected model.model.language_model.layers (Gemma-3), "
            "model.model.layers (Llama/Mistral/Qwen/Gemma), or "
            "model.transformer.h (GPT-style)."
        )
    idx = hook_layer if hook_layer >= 0 else len(layers) + hook_layer
    if not (0 <= idx < len(layers)):
        raise IndexError(
            f"crisp_unlearn: hook_layer {hook_layer} (resolved to {idx}) "
            f"out of range for model with {len(layers)} decoder layers."
        )
    return layers[idx]


def _capture_hidden(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    target_module: nn.Module,
) -> torch.Tensor:
    """Run ``model(**batch)`` and return the hidden-state tensor emitted by
    ``target_module``.

    The hook captures ``output[0]`` when the module returns a tuple (standard
    for HF decoder blocks), or ``output`` directly otherwise.  Gradients flow
    through the captured tensor — this function is called inside a grad-enabled
    context.
    """
    captured: List[torch.Tensor] = []

    def _hook(_m: nn.Module, _i: object, out: object) -> None:
        h = out[0] if isinstance(out, tuple) else out  # type: ignore[index]
        captured.append(h)  # keep grad

    handle = target_module.register_forward_hook(_hook)
    try:
        model(**batch)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(
            "crisp_unlearn: hook on target layer did not fire. "
            "Check that hook_layer points to a real decoder block."
        )
    return captured[0]


def _suppression_loss(
    h: torch.Tensor,
    sae: nn.Module,
    concept_features: torch.Tensor,
) -> torch.Tensor:
    """Compute the SAE concept-feature suppression term.

    Encodes ``h`` (shape ``(batch, seq, hidden)`` or ``(batch, hidden)``) via
    ``sae.encode(h)`` and returns::

        ‖f_concept‖₂² / batch_size

    where ``f_concept`` are the SAE feature activations at indices
    ``concept_features``.  Division by batch size keeps the scale consistent
    across variable-size batches.
    """
    # sae.encode may expect (batch, seq, hidden) or (batch, hidden); pass as-is.
    features = sae.encode(h)  # (..., n_features)

    # Index the concept sub-vector.  ``concept_features`` is a 1-D LongTensor.
    f_concept = features[..., concept_features]  # (..., n_concept)

    # Squared L2 norm over all positions and concept dims, normalised by batch.
    batch_size = h.shape[0]
    return (f_concept ** 2).sum() / float(batch_size)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def crisp_unlearn(
    model: nn.Module,
    sae: nn.Module,
    concept_features: torch.Tensor,
    forget_dataset: Iterable[Dict[str, torch.Tensor]],
    retain_dataset: Iterable[Dict[str, torch.Tensor]],
    gamma: float = 1.0,
    num_steps: int = 500,
    lr: float = 2e-5,
    hook_layer: int = -1,
    device: str = "cpu",
) -> nn.Module:
    """Permanent concept unlearning via SAE feature suppression (arXiv:2508.13650).

    Unlike inference-time SAE ablation (which requires the SAE present at every
    forward pass), CRISP fine-tunes the model to suppress target SAE features,
    making unlearning persistent and runtime-free.

    The optimisation objective is::

        L_CRISP(θ) = L_retain(θ; retain_data)
                   + γ · ‖f_concept(h_ℓ(θ; forget_data))‖₂²

    where ``h_ℓ`` are the residual-stream activations at ``hook_layer`` and
    ``f_concept`` are the SAE feature activations corresponding to
    ``concept_features``.

    Args:
        model: the causal LM to unlearn. Updated in place.
        sae: frozen sparse autoencoder.  Must expose ``.encode(h) -> features``
            where ``h`` is a (batch, [seq,] hidden) tensor.
        concept_features: 1-D ``torch.LongTensor`` of SAE feature indices to
            suppress (the features identified for the target concept).
        forget_dataset: iterable of batches on the forget (harmful) set.
            Each batch is a dict with at least ``"input_ids"``; passing
            ``"labels"`` is not required (the forget objective is purely the
            suppression term).
        retain_dataset: iterable of batches on the retain (utility) set.
            Each batch should carry ``"labels"`` for the CE retain loss.  If
            ``"labels"`` is absent the retain CE loss is skipped for that step.
        gamma: weight on the SAE suppression term (γ in the paper).
        num_steps: number of (forget, retain) batch pairs to process.
        lr: AdamW learning rate.
        hook_layer: decoder-block index to hook the SAE onto. ``-1`` = last
            layer (paper default).
        device: device to cast batches to. Ignored if the model is already on
            the right device; provided for convenience.

    Returns:
        The updated model (same object, mutated in place).
    """
    # Freeze SAE; it is never updated.
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    # Move model to device and set to train mode.
    model.to(device)
    model.train()

    # Resolve the hook target once.
    hook_module = _resolve_hook_layer(model, hook_layer)

    # All model parameters are trainable (full-weight fine-tune per the paper).
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    concept_features = concept_features.to(device)

    forget_iter = iter(forget_dataset)
    retain_iter = iter(retain_dataset)

    steps = 0
    while steps < num_steps:
        # --- Sample one batch from each dataset, cycling if necessary. ---
        try:
            forget_batch = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_dataset)
            try:
                forget_batch = next(forget_iter)
            except StopIteration:
                logger.warning("crisp_unlearn: forget_dataset is empty; stopping.")
                break

        try:
            retain_batch = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_dataset)
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                logger.warning("crisp_unlearn: retain_dataset is empty; stopping.")
                break

        # Move batches to device.
        forget_batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in forget_batch.items()
        }
        retain_batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in retain_batch.items()
        }

        optimizer.zero_grad(set_to_none=True)

        # --- Retain loss: standard CE to preserve utility. ---
        retain_loss = torch.tensor(0.0, device=device)
        if "labels" in retain_batch:
            retain_out = model(**retain_batch)
            retain_loss = retain_out.loss
        else:
            # No labels supplied — skip CE for this step.
            logger.debug("crisp_unlearn: retain batch has no 'labels'; skipping CE.")

        # --- Suppression loss: drive concept SAE features → 0 on forget set. ---
        # We need gradients through h, so capture with grad enabled.
        # Strip 'labels' from the forget batch — we do not want CE here.
        forget_fwd_batch = {k: v for k, v in forget_batch.items() if k != "labels"}
        h_forget = _capture_hidden(model, forget_fwd_batch, hook_module)
        sup_loss = gamma * _suppression_loss(h_forget, sae, concept_features)

        # --- Combined loss. ---
        loss = retain_loss + sup_loss
        loss.backward()
        optimizer.step()
        steps += 1

        logger.info(
            "crisp_unlearn: step %d/%d — loss=%.4g retain=%.4g suppression=%.4g",
            steps, num_steps,
            loss.item(),
            retain_loss.item() if isinstance(retain_loss, torch.Tensor) else retain_loss,
            sup_loss.item(),
        )

    logger.info("crisp_unlearn: completed (%d optimizer step(s)).", steps)
    return model


__all__ = ["crisp_unlearn"]
