"""MSCP-like training-free projection patch."""

from typing import Any, Dict, List

from .base import PatchState, SafetyPatch, TORCH_AVAILABLE


class MSCPProjectionPatch(SafetyPatch):
    """
    Subtract or orthogonalize weights along a safety direction vector.

    Supports both **dict mode** (``apply()``) and **PyTorch model mode**
    (``apply_to_model()``).
    """

    patch_id = "mscp_projection"

    def apply(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply to a plain-dict model state (test / lightweight mode)."""
        direction: List[float] = [float(x) for x in self.params.get("direction", [])]
        coeff = float(self.params.get("coefficient", 0.1))
        weight_key = self.params.get("weight_key", "weights")
        mode = str(self.params.get("mode", "subtract")).lower()

        weights = [float(w) for w in model_state.get(weight_key, [])]
        if not weights or not direction:
            return dict(model_state)

        n = min(len(weights), len(direction))
        original = list(weights)
        if mode == "orthogonal":
            dot = sum(weights[i] * direction[i] for i in range(n))
            norm_sq = sum(direction[i] * direction[i] for i in range(n)) or 1e-12
            proj_scale = dot / norm_sq
            for i in range(n):
                weights[i] = weights[i] - coeff * proj_scale * direction[i]
        else:
            for i in range(n):
                weights[i] = weights[i] - coeff * direction[i]

        out = dict(model_state)
        out[weight_key] = weights
        self._last_state = PatchState(
            patch_id=self.patch_id,
            metadata={"coefficient": coeff, "mode": mode, "weight_key": weight_key, **self.metadata()},
            payload={"original_values": {weight_key: original}, "direction": direction[:n]},
        )
        return out

    def apply_to_model(self, model: Any) -> None:
        """Apply MSCP projection in-place to an nn.Module.

        For each named parameter ``W`` (treating it as a flattened vector):
        - ``subtract`` mode: ``W -= coeff * d`` (broadcast subtraction)
        - ``orthogonal`` mode: ``W -= coeff * (W·d / ||d||²) * d``

        ``direction`` can be:
          - a flat list (applied uniformly to *all* parameters, truncated to
            each param's numel)
          - a dict ``{param_name: List[float]}`` for per-layer directions
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for apply_to_model(). "
                "Install torch or use apply() (dict mode) instead."
            )
        import torch as _torch

        coeff = float(self.params.get("coefficient", 0.1))
        mode = str(self.params.get("mode", "subtract")).lower()
        raw_dir = self.params.get("direction", [])

        # Support both flat list (broadcast) and per-param dict.
        per_param_dirs: Dict[str, Any] = {}
        if isinstance(raw_dir, dict):
            per_param_dirs = {k: _torch.tensor(v, dtype=_torch.float32) for k, v in raw_dir.items()}
        flat_dir = _torch.tensor(raw_dir, dtype=_torch.float32) if isinstance(raw_dir, list) else None

        self._backup_params(model)
        with _torch.no_grad():
            for name, param in model.named_parameters():
                flat_w = param.data.view(-1)
                n = flat_w.shape[0]

                if name in per_param_dirs:
                    d = per_param_dirs[name].to(param.device)
                elif flat_dir is not None and flat_dir.numel() > 0:
                    d = flat_dir[:n].to(param.device)
                else:
                    continue  # no direction for this param

                d = d[:n]
                if mode == "orthogonal":
                    dot = (flat_w[:len(d)] * d).sum()
                    norm_sq = (d * d).sum().clamp(min=1e-12)
                    flat_w[:len(d)] -= coeff * (dot / norm_sq) * d
                else:
                    flat_w[:len(d)] -= coeff * d

                param.data.copy_(flat_w.view(param.shape))
