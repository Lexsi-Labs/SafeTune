"""
SPPFT Trainer adapter.

Thin wrapper around :class:`transformers.Trainer` that implements **SPPFT**
(Safely Partial-Parameter Fine-Tuning) from:

    "Safety Layers in Aligned Large Language Models: The Key to LLM Security",
    Li, Yao, Zhang, Li — ICLR 2025, arXiv:2408.17003.
    Reference implementation: https://github.com/listen0425/Safety-Layers
    (``Code/Fine_tune/SPPFT.py``).

The paper localizes a small set of **contiguous middle layers** ("safety
layers") and SPPFT freezes the gradient of those layers' parameters while
fine-tuning the rest. In the authors' ``SPPFT.py`` the freeze is governed by
``begin_num`` / ``end_num`` (defaults 4 / 15) and applies to every
``self_attn`` and ``mlp`` submodule of layers ``begin_num < idx < end_num``
(i.e. layers 5..14) via ``param.requires_grad = False``.

This adapter reproduces that behaviour:

* The default localization method is a **contiguous middle-layer range**,
  matching the paper's SPPFT (the safety layers are middle layers, *not* the
  first layers and *not* a weight-cosine proxy). The range is taken from the
  authors' ``begin_num`` / ``end_num`` exclusive bounds, scaled to the model's
  actual depth when it differs from the reference 7B (32-layer) model.
* Optional ``aligned_state_dict`` / ``base_state_dict`` still enable the
  cosine-similarity locator from ``core.optim.safety_layers`` for callers who
  want the (weaker) weight-divergence heuristic — but it is no longer the
  default and is no longer triggered implicitly.
* Explicit ``safety_layer_indices`` on the config always take precedence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

try:
    from transformers import Trainer, TrainingArguments
    _TRAINER_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    Trainer = object  # type: ignore[assignment,misc]
    TrainingArguments = object  # type: ignore[assignment,misc]
    _TRAINER_IMPORT_ERROR = _e

try:
    from safetune.core.optim.safety_layers import (
        SafetyLayerLocator,
        SafetyLayersConfig,
        SPPFTWrapper,
    )
    _SPPFT_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    SafetyLayerLocator = None  # type: ignore[assignment]
    SafetyLayersConfig = None  # type: ignore[assignment]
    SPPFTWrapper = None  # type: ignore[assignment]
    _SPPFT_IMPORT_ERROR = _e


# Reference geometry from the authors' SPPFT.py (Llama-2-7B, 32 layers):
# `begin_num=4`, `end_num=15`, freezing layers `begin_num < idx < end_num`,
# i.e. the contiguous middle block of layers 5..14 inclusive.
_REF_NUM_LAYERS = 32
_REF_BEGIN_NUM = 4   # exclusive lower bound
_REF_END_NUM = 15    # exclusive upper bound


if _TRAINER_IMPORT_ERROR is None:
    @dataclass
    class SPPFTConfig(TrainingArguments):  # type: ignore[misc]
        """TrainingArguments with SPPFT safety-layer configuration.

        Attributes:
            num_safety_layers: kept for backward compatibility. Only used as a
                *width* hint for the contiguous middle-layer range when the model
                depth cannot be inferred; the paper's localization is range-based,
                not count-based.
            safety_layer_indices: explicit safety-layer indices; when non-empty
                these are frozen verbatim (``manual`` localization).
            sppft_mode: ``"freeze"`` (paper default — freeze safety-layer params)
                or ``"scale"`` (reduce their LR; non-paper convenience mode).
            sppft_begin_num: exclusive lower bound of the contiguous middle-layer
                range, in the authors' 32-layer reference frame. Default 4.
            sppft_end_num: exclusive upper bound of the contiguous middle-layer
                range, in the authors' 32-layer reference frame. Default 15.
                With the defaults the frozen safety layers are 5..14.
        """

        num_safety_layers: int = 8
        safety_layer_indices: List[int] = field(default_factory=list)
        sppft_mode: str = "freeze"
        sppft_begin_num: int = _REF_BEGIN_NUM
        sppft_end_num: int = _REF_END_NUM
else:  # pragma: no cover
    class SPPFTConfig(object):  # type: ignore[assignment]
        pass


def _count_decoder_layers(model: Any) -> Optional[int]:
    """Best-effort count of transformer decoder layers in ``model``."""
    # 1) HF config — the reliable path.
    cfg = getattr(model, "config", None)
    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        n = getattr(cfg, attr, None)
        if isinstance(n, int) and n > 0:
            return n
    # 2) Fall back to scanning parameter names for the max `layers.<N>` index.
    max_idx = -1
    try:
        for name, _ in model.named_parameters():
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    max_idx = max(max_idx, int(parts[i + 1]))
    except Exception:  # pragma: no cover - defensive
        return None
    return max_idx + 1 if max_idx >= 0 else None


def _middle_layer_indices(
    num_layers: Optional[int],
    begin_num: int,
    end_num: int,
    width_hint: int,
) -> List[int]:
    """Contiguous middle-layer safety-layer indices, per the paper's SPPFT.

    Reproduces the authors' ``begin_num < idx < end_num`` selection. When the
    model depth differs from the 32-layer reference the (exclusive) bounds are
    scaled proportionally so the safety block stays in the *middle* of the
    network for models of any depth.
    """
    if num_layers is None or num_layers <= 0:
        # No depth info: centre a `width_hint`-wide block on a 32-layer frame.
        num_layers = _REF_NUM_LAYERS

    if num_layers == _REF_NUM_LAYERS:
        lo, hi = begin_num, end_num
    else:
        scale = num_layers / _REF_NUM_LAYERS
        lo = int(round(begin_num * scale))
        hi = int(round(end_num * scale))

    # `begin_num < idx < end_num` -> first frozen index lo+1, last hi-1.
    start = lo + 1
    stop = hi  # exclusive
    indices = [i for i in range(start, stop) if 0 <= i < num_layers]

    if not indices:
        # Degenerate bounds: fall back to a `width_hint`-wide centred block.
        width = max(1, min(width_hint, num_layers))
        start = max(0, (num_layers - width) // 2)
        indices = list(range(start, min(num_layers, start + width)))
    return indices


class SPPFTTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """Trainer subclass that freezes safety-critical layers before training.

    By default the safety layers are the **contiguous middle layers** of the
    network (the paper's localization result for SPPFT); their ``self_attn``
    and ``mlp`` parameters have ``requires_grad`` cleared so only the remaining
    layers are fine-tuned.

    Args:
        aligned_state_dict: optional state dict of the aligned model. When
            supplied *together with* ``base_state_dict`` the cosine-similarity
            weight-divergence locator is used instead of the middle-layer
            range. This is a non-paper heuristic, off by default.
        base_state_dict: optional state dict of the base (pre-alignment) model.
        safety_layer_indices: optional explicit safety-layer indices; when
            given these override every other localization method.
    """

    def __init__(
        self,
        *args: Any,
        aligned_state_dict: Optional[Dict[str, Any]] = None,
        base_state_dict: Optional[Dict[str, Any]] = None,
        safety_layer_indices: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for SPPFTTrainer"
            ) from _TRAINER_IMPORT_ERROR
        if _SPPFT_IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.optim.safety_layers is unavailable"
            ) from _SPPFT_IMPORT_ERROR

        super().__init__(*args, **kwargs)

        # Explicit indices: kwarg first, then config field.
        explicit_indices = list(
            safety_layer_indices
            if safety_layer_indices is not None
            else (getattr(self.args, "safety_layer_indices", []) or [])
        )

        num_safety = int(getattr(self.args, "num_safety_layers", 8))
        begin_num = int(getattr(self.args, "sppft_begin_num", _REF_BEGIN_NUM))
        end_num = int(getattr(self.args, "sppft_end_num", _REF_END_NUM))

        # Localization precedence:
        #   1. explicit indices                       -> manual
        #   2. aligned + base state dicts supplied     -> cosine (non-paper)
        #   3. default                                 -> middle-layer range
        use_cosine = (
            not explicit_indices
            and aligned_state_dict is not None
            and base_state_dict is not None
        )

        if explicit_indices:
            method = "manual"
            manual_indices = explicit_indices
        elif use_cosine:
            method = "cosine"
            manual_indices = []
        else:
            # Paper-faithful default: contiguous *middle* layers.
            method = "manual"
            num_layers = _count_decoder_layers(self.model)
            manual_indices = _middle_layer_indices(
                num_layers, begin_num, end_num, width_hint=num_safety
            )

        layers_cfg = SafetyLayersConfig(
            localization_method=method,
            safety_layer_indices=manual_indices,
            sppft_mode=getattr(self.args, "sppft_mode", "freeze"),
        )
        locator = SafetyLayerLocator(layers_cfg)
        safety_layers: Set[int] = locator.locate(
            aligned_state_dict=aligned_state_dict,
            base_state_dict=base_state_dict,
        )
        self.safety_layers: Set[int] = set(safety_layers)
        self._sppft = SPPFTWrapper(
            model=self.model, safety_layers=safety_layers, config=layers_cfg
        )
        # Freeze safety-layer params by setting requires_grad=False, exactly as
        # the authors' SPPFT.py does (`param.requires_grad = False`).
        self._sppft.apply()

    def train(self, *args, **kwargs):
        """Run training and always restore frozen safety-layer parameters afterwards.

        ``_sppft.apply()`` freezes safety-layer params at ``__init__``. Without
        this override the frozen state leaks out to the caller after training,
        leaving the model with parameters that silently have ``requires_grad=False``.
        The ``finally`` block guarantees restore() runs even if training raises.
        """
        try:
            return super().train(*args, **kwargs)
        finally:
            self._sppft.restore()

    def training_step(self, model, inputs, num_items_in_batch=None):  # type: ignore[override]
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except TypeError:
            return super().training_step(model, inputs)
