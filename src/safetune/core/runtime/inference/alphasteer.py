"""AlphaSteer (LEGACY runtime/inference implementation).

⚠️ SUPERSEDED, kept for back-compat. For the maintained, paper-checked
AlphaSteer (closed-form null-space-constrained steering matrix) use
:class:`safetune.steer.alphasteer.AlphaSteerModel`; this legacy module is no
longer the canonical implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch


@dataclass
class AlphaSteerConfig:
    layer_id: int = 15
    lambda_ridge: float = 1e-4
    null_rank: Optional[int] = None


class AlphaSteerWrapper:
    def __init__(self, model: Any, config: Optional[AlphaSteerConfig] = None) -> None:
        self.model = model
        self.config = config or AlphaSteerConfig()
        self.M: Optional[torch.Tensor] = None
        self._hooks: List[Any] = []

    def fit(
        self,
        harmful_activations: torch.Tensor,
        benign_activations: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        X_h = harmful_activations.float()
        X_b = benign_activations.float()
        d = X_h.shape[1]

        if targets is None:
            Y = -X_h
        else:
            Y = targets.float()

        U, S, Vt = torch.linalg.svd(X_b, full_matrices=False)
        if self.config.null_rank is not None:
            k = int(self.config.null_rank)
        else:
            k = int((S > 1e-6).sum().item())
        k = max(0, min(k, Vt.shape[0]))
        row_space_basis = Vt[:k]
        I_d = torch.eye(d, dtype=X_h.dtype, device=X_h.device)
        if k > 0:
            null_proj = I_d - row_space_basis.T @ row_space_basis
        else:
            null_proj = I_d

        A = X_h.T @ X_h + self.config.lambda_ridge * I_d
        B = X_h.T @ Y
        M_unconstrained = torch.linalg.solve(A, B)
        M = null_proj @ M_unconstrained
        self.M = M.detach().to(torch.float32)
        return self.M

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
            if self.M is None:
                return output
            if isinstance(output, tuple):
                hidden = output[0]
                M = self.M.to(hidden.device).to(hidden.dtype)
                delta = hidden @ M.T
                hidden = hidden + delta
                return (hidden,) + output[1:]
            else:
                M = self.M.to(output.device).to(output.dtype)
                return output + output @ M.T

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

    def __enter__(self) -> "AlphaSteerWrapper":
        self.register_hooks()
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()
