"""SafeSteer (LEGACY runtime/inference implementation).

⚠️ SUPERSEDED, kept for back-compat. For the maintained SafeSteer (category-wise
diff-of-means activation steering, with the optional per-prompt category
router) use :class:`safetune.steer.safesteer.SafeSteerModel`; this legacy
module is no longer the canonical implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch


@dataclass
class SafeSteerConfig:
    layer_id: int = 20
    alpha: float = 1.0
    default_category: str = "default"


class SafeSteerWrapper:
    def __init__(
        self,
        model: Any,
        category_vectors: Dict[str, torch.Tensor],
        classifier: Optional[Callable[[str], str]] = None,
        config: Optional[SafeSteerConfig] = None,
    ) -> None:
        self.model = model
        self.category_vectors = category_vectors
        self.classifier = classifier
        self.config = config or SafeSteerConfig()
        self._hooks: List[Any] = []
        self._current_category: Optional[str] = None

    @classmethod
    def compute_category_vectors(
        cls,
        safe_acts: Dict[str, torch.Tensor],
        unsafe_acts: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        vectors: Dict[str, torch.Tensor] = {}
        for cat in safe_acts.keys():
            if cat not in unsafe_acts:
                continue
            s = safe_acts[cat]
            u = unsafe_acts[cat]
            vectors[cat] = s.mean(dim=0) - u.mean(dim=0)
        return vectors

    def _get_layer(self) -> Any:
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "layers"):
            return self.model.language_model.layers[self.config.layer_id]
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[self.config.layer_id]
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[self.config.layer_id]
        raise AttributeError(
            "Could not locate transformer layers on model "
            "(tried language_model.layers, model.layers, transformer.h)"
        )

    def _hook_fn(self) -> Callable:
        def hook(module: Any, inputs: Any, output: Any) -> Any:
            cat = self._current_category or self.config.default_category
            if cat not in self.category_vectors:
                return output
            v = self.category_vectors[cat]
            if isinstance(output, tuple):
                hidden = output[0]
                add = (self.config.alpha * v).to(hidden.device).to(hidden.dtype)
                hidden = hidden + add[None, None, :]
                return (hidden,) + output[1:]
            else:
                add = (self.config.alpha * v).to(output.device).to(output.dtype)
                return output + add[None, None, :]

        return hook

    def register_hooks(self) -> None:
        self.remove_hooks()
        layer = self._get_layer()
        h = layer.register_forward_hook(self._hook_fn())
        self._hooks.append(h)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def set_current_prompt(self, prompt_text: str) -> None:
        if self.classifier is None:
            self._current_category = self.config.default_category
            return
        try:
            self._current_category = self.classifier(prompt_text)
        except Exception:
            self._current_category = self.config.default_category

    def __enter__(self) -> "SafeSteerWrapper":
        self.register_hooks()
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()
