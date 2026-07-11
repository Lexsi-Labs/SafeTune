"""
SCANS (LEGACY runtime/inference implementation).

⚠️ SUPERSEDED, simplified version kept for back-compat. It computes a single
diff-of-means steering vector (`mean(safe) − mean(unsafe)`) applied
unconditionally — it does NOT implement SCANS's defining mechanism (the
per-prompt transition-point classifier that conditionally signs/gates steering
to mitigate over-refusal). For the faithful SCANS (zouyingcao/SCANS, AAAI 2025)
use :class:`safetune.steer.scans.SCANSModel`. Do not cite this legacy variant
as SCANS.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SCANSConfig:
    """Configuration for SCANS activation steering."""
    # Layers to apply steering (default: layers 10-19)
    target_layers: List[int] = field(default_factory=lambda: list(range(10, 20)))
    # Steering multiplier α
    multiplier: float = 3.5
    # Size of anchor dataset for computing steering vectors
    anchor_size: int = 64


class SCANSWrapper:
    """
    SCANS: computes steering + reference vectors and applies them during inference.

    Usage::

        wrapper = SCANSWrapper(model, config)
        wrapper.compute_vectors(safe_inputs, unsafe_inputs)
        wrapper.register_hooks()
        output = model.generate(input_ids)
        wrapper.remove_hooks()
    """

    def __init__(
        self,
        model: Any,
        config: Optional[SCANSConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or SCANSConfig()
        self._steering_vectors: Dict[int, Any] = {}
        self._hooks: List[Any] = []

    def compute_vectors(
        self,
        safe_activations: Dict[int, Any],
        unsafe_activations: Dict[int, Any],
    ) -> None:
        """
        Compute steering vector = mean(safe) - mean(unsafe) per layer.

        Args:
            safe_activations: {layer_idx: tensor(n_samples, hidden_dim)}
            unsafe_activations: {layer_idx: tensor(n_samples, hidden_dim)}
        """
        for layer_idx in self.config.target_layers:
            if layer_idx in safe_activations and layer_idx in unsafe_activations:
                safe_mean = safe_activations[layer_idx].float().mean(dim=0)
                unsafe_mean = unsafe_activations[layer_idx].float().mean(dim=0)
                self._steering_vectors[layer_idx] = safe_mean - unsafe_mean
        logger.info("SCANS: computed steering vectors for %d layers.", len(self._steering_vectors))

    def set_precomputed_vectors(self, vectors: Dict[int, Any]) -> None:
        """Load precomputed steering vectors."""
        self._steering_vectors = vectors

    def _hook_fn(self, layer_idx: int):
        def hook(module: Any, input: Any, output: Any) -> Any:
            if layer_idx not in self._steering_vectors:
                return output
            sv = self._steering_vectors[layer_idx].to(output[0].device).to(output[0].dtype)
            if isinstance(output, tuple):
                hidden = output[0] + self.config.multiplier * sv
                return (hidden,) + output[1:]
            return output + self.config.multiplier * sv
        return hook

    def _get_layers(self) -> list:
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "layers"):
            return list(self.model.language_model.layers)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        return []

    def register_hooks(self) -> None:
        self.remove_hooks()
        layers = self._get_layers()
        for idx in self.config.target_layers:
            if idx < len(layers) and idx in self._steering_vectors:
                h = layers[idx].register_forward_hook(self._hook_fn(idx))
                self._hooks.append(h)
        logger.info("SCANS: registered hooks on %d layers.", len(self._hooks))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, *args):
        self.remove_hooks()
