"""
RMU: Representation Misdirection for Unlearning (Li et al.,
"The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning",
2024, arXiv:2403.03218).
Authors' reference implementation:
https://github.com/centerforaisafety/wmdp -- ``rmu/unlearn.py`` (``run_rmu``)
and ``rmu/utils.py`` (``get_params``, ``forward_with_cache``).

Algorithm (faithful to the authors' ``run_rmu`` loop):

  RMU edits a small set of early-layer MLP parameters so the model's
  residual-stream activations at a chosen *target layer* are pushed onto a
  fixed random "control" direction on the forget set, while staying close
  to a frozen reference model on the retain set.

  * Setup. A frozen deepcopy ``frozen_model`` of the trained model is the
    reference. The activations are read from a single decoder block -- the
    output of ``layers[layer_id]`` -- via a forward hook (the authors'
    ``forward_with_cache``: it caches ``output[0]`` of the module). Only the
    parameters selected by ``get_params`` are trainable -- in the authors'
    config those are the MLP ``down_proj`` weights of layers ``5,6,7``
    (``param_ids = [6]`` indexing into ``model.model.layers[l].parameters()``).
    SafeTune selects them by name (``mlp.down_proj``) which is robust across
    HF model variants.

  * Control vector. For each forget "topic" a fixed random unit vector is
    drawn once and scaled by ``steering_coeff``::

        random_vector = torch.rand(1, 1, hidden_size)
        control_vec   = random_vector / ||random_vector|| * steering_coeff

  * Forget step. On a forget batch, run the *updated* model with grad and
    cache the target-layer activations ``act_updated``::

        forget_loss = MSE(act_updated, control_vec)

    This drives the forget activations toward the random control direction
    ("misdirection").

  * Retain step. On a retain batch, cache the target-layer activations of
    both the updated model (with grad) and the frozen model (no grad)::

        retain_loss = alpha * MSE(act_updated, act_frozen)

  * Update. ``loss = forget_loss + retain_loss``; one AdamW step per batch.
    The authors run a single epoch over ``num_batches`` batches.

For safety unlearning we set:
  * F = harmful examples whose internal representations we want to scramble.
  * R = benign examples whose representations we want to keep unchanged.

The public entry point ``rmu_unlearn`` takes a model, a frozen reference (or
None to snapshot the model itself), iterables over retain / forget batches,
and runs the RMU loop in place.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetune._refusal_helpers import _get_decoder_layers

logger = logging.getLogger(__name__)


@dataclass
class RMUConfig:
    """Configuration for RMU unlearning.

    Attributes:
        layer_id: index of the decoder block whose *output* residual-stream
            activations are steered (the authors' ``--layer_id``, default
            ``7``).
        update_layer_ids: decoder-block indices whose MLP ``down_proj``
            parameters are made trainable (the authors' ``--layer_ids``,
            default ``[5, 6, 7]`` -- the target layer and the two below it).
        steering_coeff: scalar ``c`` that scales the random control unit
            vector (the authors' ``--steering_coeffs``, default ``20.0``).
        alpha: weight on the retain MSE term (the authors' ``--alpha``,
            default ``100.0``).
        lr: AdamW learning rate (the authors' ``--lr``, default ``5e-5``).
        max_num_batches: hard cap on the number of (forget, retain) batch
            pairs consumed (the authors' ``--max_num_batches``, default
            ``80``). ``None`` consumes every available pair.
        param_substring: substring identifying which named parameters of a
            decoder block are trainable. ``"mlp.down_proj"`` reproduces the
            authors' ``param_ids = [6]`` selection in a name-based,
            architecture-robust way.
    """

    layer_id: int = 7
    update_layer_ids: List[int] = None  # type: ignore[assignment]
    steering_coeff: float = 20.0
    alpha: float = 100.0
    lr: float = 5e-5
    max_num_batches: Optional[int] = 80
    param_substring: str = "mlp.down_proj"

    def __post_init__(self) -> None:
        if self.update_layer_ids is None:
            # Authors' default: target layer and the two below it.
            self.update_layer_ids = [
                self.layer_id - 2,
                self.layer_id - 1,
                self.layer_id,
            ]


def _select_params(model: nn.Module, cfg: RMUConfig) -> List[nn.Parameter]:
    """Return the trainable parameter list (authors' ``get_params``).

    Selects parameters whose qualified name lies inside one of the
    ``update_layer_ids`` decoder blocks and contains ``param_substring``
    (``mlp.down_proj`` -- the MLP output projection). All other parameters
    have ``requires_grad`` cleared.
    """
    layers = _get_decoder_layers(model)
    if not layers:
        raise ValueError(
            "rmu_unlearn: could not locate decoder layers on model. "
            "Expected model.model.language_model.layers (Gemma-3), "
            "model.model.layers (Llama/Mistral/Qwen/Gemma), or "
            "model.transformer.h (GPT-style)."
        )

    wanted = set()
    for lid in cfg.update_layer_ids:
        if lid < 0 or lid >= len(layers):
            raise ValueError(
                f"rmu_unlearn: update layer id {lid} out of range "
                f"(model has {len(layers)} decoder layers)."
            )
        block = layers[lid]
        for name, p in block.named_parameters():
            if cfg.param_substring in name:
                wanted.add(id(p))

    params: List[nn.Parameter] = []
    for p in model.parameters():
        if id(p) in wanted:
            p.requires_grad_(True)
            params.append(p)
        else:
            p.requires_grad_(False)

    if not params:
        raise ValueError(
            f"rmu_unlearn: no parameters matched substring "
            f"'{cfg.param_substring}' in layers {cfg.update_layer_ids}."
        )
    return params


def _forward_with_cache(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    module: nn.Module,
    *,
    no_grad: bool,
) -> torch.Tensor:
    """Run ``model`` and return the cached output of ``module``.

    Faithful to the authors' ``forward_with_cache``: a forward hook on
    ``module`` records ``output[0]`` (if the module returns a tuple, e.g. a
    decoder block) or ``output`` directly, and that tensor is returned.
    """
    cache: List[torch.Tensor] = []

    def hook(_mod, _inp, output):  # noqa: ANN001
        cache.append(output[0] if isinstance(output, tuple) else output)
        return None

    handle = module.register_forward_hook(hook)
    # Strip labels: RMU only reads activations via hook, never uses the
    # loss output. Passing labels causes HF to run an extra loss computation.
    fwd_batch = {k: v for k, v in batch.items() if k != "labels"}
    try:
        if no_grad:
            with torch.no_grad():
                model(**fwd_batch)
        else:
            model(**fwd_batch)
    finally:
        handle.remove()

    if not cache:
        raise RuntimeError("rmu_unlearn: target module did not produce activations.")
    return cache[0]


def rmu_unlearn(
    model: nn.Module,
    retain_batches: Iterable[Dict[str, torch.Tensor]],
    forget_batches: Iterable[Dict[str, torch.Tensor]],
    *,
    frozen_model: Optional[nn.Module] = None,
    config: Optional[RMUConfig] = None,
) -> nn.Module:
    """Run RMU unlearning in place on ``model``.

    Faithful to the authors' ``run_rmu`` loop: a single random control
    vector is drawn and scaled by ``steering_coeff``; then for each paired
    ``(forget_batch, retain_batch)`` one AdamW step is taken on::

        loss = MSE(act_updated_forget, control_vec)
             + alpha * MSE(act_updated_retain, act_frozen_retain)

    where ``act_*`` are the output activations of decoder block
    ``layer_id`` captured by a forward hook. Only the MLP ``down_proj``
    parameters of ``update_layer_ids`` are updated.

    Args:
        model: the model to unlearn. Updated in place. Its
            ``requires_grad`` flags are reconfigured by :func:`_select_params`.
        retain_batches: iterable yielding batches of retain examples (kwargs
            for ``model(**batch)``, e.g. ``input_ids`` / ``attention_mask``).
        forget_batches: iterable yielding batches of forget examples. The
            two iterables are zipped; the loop stops at the shorter one (or
            at ``max_num_batches``).
        frozen_model: frozen reference; if ``None``, a deepcopy of ``model``
            is taken once at start (before parameter selection, so it keeps
            the original weights).
        config: :class:`RMUConfig`.

    Returns:
        The updated model (same object, mutated in place).
    """
    cfg = config or RMUConfig()

    # Snapshot the frozen reference BEFORE selecting params / training so it
    # holds the original weights.
    if frozen_model is None:
        frozen_model = copy.deepcopy(model)
    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad_(False)

    model.train()
    params = _select_params(model, cfg)
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)

    updated_layers = _get_decoder_layers(model)
    frozen_layers = _get_decoder_layers(frozen_model)
    if cfg.layer_id < 0 or cfg.layer_id >= len(updated_layers):
        raise ValueError(
            f"rmu_unlearn: layer_id {cfg.layer_id} out of range "
            f"(model has {len(updated_layers)} decoder layers)."
        )
    updated_module = updated_layers[cfg.layer_id]
    frozen_module = frozen_layers[cfg.layer_id]

    # Fixed random control vector, drawn once (authors' control_vec).
    ref_param = params[0]
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is None:
        # Fall back to the down_proj output dim (== hidden size).
        hidden_size = ref_param.shape[0]
    random_vector = torch.rand(
        1, 1, hidden_size, dtype=ref_param.dtype, device=ref_param.device
    )
    control_vec = random_vector / torch.norm(random_vector) * cfg.steering_coeff

    cap = cfg.max_num_batches
    steps = 0
    for forget_batch, retain_batch in zip(forget_batches, retain_batches):
        if cap is not None and steps >= cap:
            break

        # --- Forget step: push activations onto the control direction. ---
        act_updated_forget = _forward_with_cache(
            model, forget_batch, updated_module, no_grad=False
        )
        forget_loss = F.mse_loss(act_updated_forget, control_vec.expand_as(act_updated_forget).to(act_updated_forget.dtype))

        # --- Retain step: stay close to the frozen reference. ---
        act_updated_retain = _forward_with_cache(
            model, retain_batch, updated_module, no_grad=False
        )
        act_frozen_retain = _forward_with_cache(
            frozen_model, retain_batch, frozen_module, no_grad=True
        )
        retain_loss = cfg.alpha * F.mse_loss(
            act_updated_retain, act_frozen_retain.to(act_updated_retain.dtype)
        )

        loss = forget_loss + retain_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps += 1

        logger.info(
            "rmu_unlearn: step %d -- loss=%.4g forget=%.4g retain=%.4g",
            steps,
            loss.item(),
            forget_loss.item(),
            retain_loss.item(),
        )

    logger.info("rmu_unlearn: completed (%d optimizer step(s)).", steps)
    return model


__all__ = ["RMUConfig", "rmu_unlearn"]