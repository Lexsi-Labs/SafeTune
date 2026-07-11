"""
AdaSteer (LEGACY runtime/inference implementation).

⚠️ SUPERSEDED, simplified version kept for back-compat: it applies a single
precomputed vector with a fixed multiplier (plain fixed-coefficient CAA) and
does NOT implement AdaSteer's two-direction adaptive R-Law/H-Law coefficient
mechanism. Use :class:`safetune.steer.adasteer.AdaSteerModel` instead. Do not
cite this legacy variant as AdaSteer (MuyuenLP/AdaSteer, EMNLP 2025).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AdaSteerConfig:
    """Configuration for AdaSteer activation steering."""
    # Layer indices at which to apply steering
    target_layers: List[int] = field(default_factory=lambda: list(range(10, 20)))
    # Base multiplier for the steering vector
    base_multiplier: float = 3.0
    # Whether to adaptively scale the multiplier based on safety signal
    adaptive: bool = True
    # Safety signal threshold: below this, steering is applied more aggressively
    safety_threshold: float = 0.5


class AdaSteerWrapper:
    """
    Inference-time activation steering using precomputed safety direction vectors.

    Usage::

        wrapper = AdaSteerWrapper(model, safety_vectors, config)
        wrapper.register_hooks()
        output = model.generate(input_ids)
        wrapper.remove_hooks()
    """

    def __init__(
        self,
        model: Any,
        safety_vectors: Dict[int, Any],
        config: Optional[AdaSteerConfig] = None,
    ) -> None:
        self.model = model
        self.safety_vectors = safety_vectors  # {layer_idx: tensor(hidden_dim,)}
        self.config = config or AdaSteerConfig()
        self._hooks: List[Any] = []
        self._current_multiplier = self.config.base_multiplier

    def _hook_fn(self, layer_idx: int):
        def hook(module: Any, input: Any, output: Any) -> Any:
            try:
                import torch
            except ImportError:
                return output

            if layer_idx not in self.safety_vectors:
                return output

            sv = self.safety_vectors[layer_idx].to(output[0].device).to(output[0].dtype)

            if isinstance(output, tuple):
                hidden = output[0]
                hidden = hidden + self._current_multiplier * sv
                return (hidden,) + output[1:]
            else:
                return output + self._current_multiplier * sv

        return hook

    def register_hooks(self) -> None:
        """Register forward hooks on target layers."""
        self.remove_hooks()
        layers = self._get_layers()
        for idx in self.config.target_layers:
            if idx < len(layers):
                h = layers[idx].register_forward_hook(self._hook_fn(idx))
                self._hooks.append(h)
        logger.info("AdaSteer: registered hooks on %d layers.", len(self._hooks))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _get_layers(self) -> list:
        """Get the list of transformer layers from the model."""
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "layers"):
            return list(self.model.language_model.layers)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        else:
            return []

    def set_adaptive_multiplier(self, safety_score: float) -> None:
        """
        Adaptively adjust the steering multiplier based on a safety score.
        Lower safety → stronger steering.
        """
        if self.config.adaptive:
            if safety_score < self.config.safety_threshold:
                scale = 1.0 + (self.config.safety_threshold - safety_score)
            else:
                scale = max(0.1, 1.0 - (safety_score - self.config.safety_threshold))
            self._current_multiplier = self.config.base_multiplier * scale
        else:
            self._current_multiplier = self.config.base_multiplier

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, *args):
        self.remove_hooks()
