"""NLSR-style neuron-level transplant patch."""

from typing import Any, Dict, List

from .base import PatchState, SafetyPatch, TORCH_AVAILABLE


class NLSRPatch(SafetyPatch):
    """
    NLSR: transplant safety-relevant neuron values from a donor model.

    Supports both **dict mode** (``apply()``) and **PyTorch model mode**
    (``apply_to_model()``).
    """

    patch_id = "nlsr"

    def apply(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply to plain-dict model state (test / lightweight mode)."""
        unit_ids: List[str] = list(self.params.get("unit_ids", []))
        donor_values: Dict[str, float] = {
            str(k): float(v) for k, v in dict(self.params.get("donor_values", {})).items()
        }
        units_key = self.params.get("units_key", "neurons")
        blend = float(self.params.get("blend", 1.0))

        neurons = dict(model_state.get(units_key, {}))
        original_values = {uid: neurons.get(uid) for uid in unit_ids if uid in neurons}
        for uid in unit_ids:
            if uid in donor_values:
                old_val = float(neurons.get(uid, 0.0))
                donor = float(donor_values[uid])
                neurons[uid] = (1.0 - blend) * old_val + blend * donor

        out = dict(model_state)
        out[units_key] = neurons
        self._last_state = PatchState(
            patch_id=self.patch_id,
            metadata={
                "units_key": units_key,
                "unit_count": len(unit_ids),
                "blend": blend,
                **self.metadata(),
            },
            payload={
                "original_values": {units_key: dict(model_state.get(units_key, {}))},
                "updated": original_values,
            },
        )
        return out

    def apply_to_model(self, model: Any) -> None:
        """Apply NLSR neuron transplanting in-place to an nn.Module.

        ``unit_ids`` format: ``"layer_name.neuron_index"`` e.g.
        ``"model.layers.3.mlp.gate_proj.weight.42"``.

        ``donor_param_path`` (str): path to a HuggingFace checkpoint from which
        to read donor values. If not provided, ``donor_values`` (dict
        ``param_name -> flat_index -> float``) must be supplied.

        ``blend`` (float, 0-1): linear interpolation weight for transplant.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for apply_to_model(). "
                "Install torch or use apply() (dict mode) instead."
            )
        import torch as _torch

        blend = float(self.params.get("blend", 1.0))
        donor_param_path = self.params.get("donor_param_path")
        # Format: {param_name: {flat_index: donor_value, ...}, ...}
        donor_map: Dict[str, Dict[int, float]] = self.params.get("donor_map", {})

        # Optionally load donor weights from a checkpoint.
        donor_state: Dict[str, Any] = {}
        if donor_param_path:
            try:
                raw = _torch.load(donor_param_path, map_location="cpu", weights_only=True)
                # Support state_dict or dict with "model" key.
                donor_state = raw.get("model", raw) if isinstance(raw, dict) else raw
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "NLSRPatch: could not load donor from %s — %s", donor_param_path, exc
                )

        self._backup_params(model)
        with _torch.no_grad():
            for name, param in model.named_parameters():
                flat_w = param.data.view(-1)

                # Use loaded donor state if available.
                if donor_state and name in donor_state:
                    donor_tensor = donor_state[name].to(param.device).view(-1)
                    n = min(flat_w.numel(), donor_tensor.numel())
                    flat_w[:n] = (1.0 - blend) * flat_w[:n] + blend * donor_tensor[:n]
                    param.data.copy_(flat_w.view(param.shape))
                    continue

                # Fall back to per-index donor_map.
                if name in donor_map:
                    for idx_str, val in donor_map[name].items():
                        idx = int(idx_str)
                        if 0 <= idx < flat_w.numel():
                            old = flat_w[idx].item()
                            flat_w[idx] = (1.0 - blend) * old + blend * float(val)
                    param.data.copy_(flat_w.view(param.shape))
