"""
EMA Callback adapter.

Thin :class:`transformers.TrainerCallback` wrapper around
``safetune.core.optim.ema.EMAOptimizerWrapper``.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from transformers import TrainerCallback
    _CB_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    TrainerCallback = object  # type: ignore[assignment,misc]
    _CB_IMPORT_ERROR = _e

try:
    from safetune.core.optim.ema import EMAOptimizerWrapper
    _EMA_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    EMAOptimizerWrapper = None  # type: ignore[assignment]
    _EMA_IMPORT_ERROR = _e


class EMACallback(TrainerCallback if _CB_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """TrainerCallback that maintains an EMA of model weights during training."""

    def __init__(self, decay: float = 0.995, device: Any = None) -> None:
        if _CB_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for EMACallback"
            ) from _CB_IMPORT_ERROR
        if _EMA_IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.optim.ema is unavailable"
            ) from _EMA_IMPORT_ERROR
        self.decay = decay
        self.device = device
        self._ema: Optional[EMAOptimizerWrapper] = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):  # type: ignore[override]
        if model is not None and self._ema is None:
            self._ema = EMAOptimizerWrapper(model, decay=self.decay, device=self.device)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):  # type: ignore[override]
        if self._ema is not None and model is not None:
            self._ema.step(model)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):  # type: ignore[override]
        if self._ema is not None and model is not None:
            self._ema.apply_ema_weights(model)
        return control
