"""
STA: Steering Target Atoms — Precise Behavior Control.
zjunlp/steer-target-atoms — ACL 2025

Instead of steering full activation layers, STA identifies minimal "target atoms"
(specific attention head outputs) and steers only through those atoms. More
precise and less disruptive than full-layer steering.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class STAConfig:
    """Configuration for STA steering."""
    # Target atoms: list of (layer_idx, head_idx) tuples
    target_atoms: List[Tuple[int, int]] = field(default_factory=list)
    # Steering multiplier
    multiplier: float = 3.0


class STAWrapper:
    """
    Steer model behavior through identified target atoms.

    Usage::

        config = STAConfig(target_atoms=[(12, 5), (14, 3)], multiplier=2.5)
        wrapper = STAWrapper(model, atom_vectors, config)
        wrapper.register_hooks()
        output = model.generate(input_ids)
        wrapper.remove_hooks()
    """

    def __init__(
        self,
        model: Any,
        atom_vectors: Dict[Tuple[int, int], Any],
        config: Optional[STAConfig] = None,
    ) -> None:
        self.model = model
        self.atom_vectors = atom_vectors  # {(layer, head): tensor(head_dim,)}
        self.config = config or STAConfig()
        self._hooks: List[Any] = []

    def _get_layers(self) -> list:
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "layers"):
            return list(self.model.language_model.layers)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        return []

    def _hook_fn(self, layer_idx: int):
        """Hook that modifies specific attention head outputs."""
        relevant_heads = {
            h: self.atom_vectors[(layer_idx, h)]
            for (l, h) in self.atom_vectors if l == layer_idx
        }

        def hook(module: Any, input: Any, output: Any) -> Any:
            if not relevant_heads:
                return output

            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # hidden shape: (batch, seq_len, hidden_dim)
            # We apply per-head steering by slicing the hidden dim
            try:
                num_heads = len(relevant_heads)
                head_dim = hidden.shape[-1] // max(num_heads, 1)

                for head_idx, sv in relevant_heads.items():
                    sv = sv.to(hidden.device).to(hidden.dtype)
                    start = head_idx * head_dim
                    end = start + sv.shape[0]
                    if end <= hidden.shape[-1]:
                        hidden[:, :, start:end] = hidden[:, :, start:end] + self.config.multiplier * sv
            except Exception as e:
                logger.debug("STA hook error: %s", e)

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        return hook

    def register_hooks(self) -> None:
        self.remove_hooks()
        layers = self._get_layers()
        affected_layers = {l for (l, h) in self.atom_vectors}
        for idx in affected_layers:
            if idx < len(layers):
                h = layers[idx].register_forward_hook(self._hook_fn(idx))
                self._hooks.append(h)
        logger.info("STA: registered hooks on %d layers.", len(self._hooks))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, *args):
        self.remove_hooks()
