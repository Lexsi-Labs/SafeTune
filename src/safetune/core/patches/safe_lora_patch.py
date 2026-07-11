"""SafeLoRA-style LoRA adapter merge patch."""

from typing import Any, Dict

from .base import PatchState, SafetyPatch, TORCH_AVAILABLE


class SafeLoRAPatch(SafetyPatch):
    """
    Merge two LoRA adapters (base + aligned) with a configurable alpha.

    Supports both **dict mode** (``apply()``) and **PyTorch model mode**
    (``apply_to_model()``).
    """

    patch_id = "safe_lora"

    def apply(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply to plain-dict adapter state (test / lightweight mode)."""
        base_adapter: Dict[str, float] = dict(self.params.get("base_adapter", {}))
        aligned_adapter: Dict[str, float] = dict(self.params.get("aligned_adapter", {}))
        alpha = float(self.params.get("alpha", 0.5))
        max_delta_norm = self.params.get("max_delta_norm")
        lora_key = self.params.get("lora_key", "lora_adapter")

        merged = {}
        all_keys = set(base_adapter) | set(aligned_adapter)
        for k in all_keys:
            b = float(base_adapter.get(k, 0.0))
            a = float(aligned_adapter.get(k, 0.0))
            merged[k] = (1.0 - alpha) * b + alpha * a

        if max_delta_norm is not None:
            try:
                import math
                norm = math.sqrt(sum((merged[k] - float(base_adapter.get(k, 0.0))) ** 2 for k in merged))
                if norm > float(max_delta_norm):
                    scale = float(max_delta_norm) / norm
                    merged = {
                        k: float(base_adapter.get(k, 0.0)) + scale * (merged[k] - float(base_adapter.get(k, 0.0)))
                        for k in merged
                    }
            except Exception:
                pass

        out = dict(model_state)
        out[lora_key] = merged
        self._last_state = PatchState(
            patch_id=self.patch_id,
            metadata={"alpha": alpha, "max_delta_norm": max_delta_norm, **self.metadata()},
            payload={"original_values": {lora_key: dict(model_state.get(lora_key, {}))}},
        )
        return out

    def apply_to_model(self, model: Any) -> None:
        """Apply SafeLoRA adapter merge in-place to an nn.Module.

        Loads two PEFT adapters (``base_adapter_path`` and
        ``aligned_adapter_path``) and merges the aligned delta into the model
        with ``alpha`` interpolation.  Requires ``peft`` to be installed.

        Alternatively, supply ``aligned_state_dict_path`` (a raw
        ``torch.save()`` state dict) to bypass peft and merge directly.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for apply_to_model(). "
                "Install torch or use apply() (dict mode) instead."
            )
        import torch as _torch

        alpha = float(self.params.get("alpha", 0.5))
        max_delta_norm = self.params.get("max_delta_norm")
        aligned_state_dict_path = self.params.get("aligned_state_dict_path")
        # In-memory state-dict alternative (avoids a save / load round-trip).
        aligned_state_dict = self.params.get("aligned_state_dict")
        base_state_dict = self.params.get("base_state_dict")

        if aligned_state_dict is not None or aligned_state_dict_path:
            # Raw state-dict merge: aligned_weights interpolated with base.
            if aligned_state_dict is None:
                try:
                    aligned_sd = _torch.load(aligned_state_dict_path, map_location="cpu", weights_only=True)
                    aligned_sd = aligned_sd.get("model", aligned_sd) if isinstance(aligned_sd, dict) else aligned_sd
                except Exception as exc:
                    raise RuntimeError(f"SafeLoRAPatch: could not load aligned state dict: {exc}") from exc
            else:
                aligned_sd = dict(aligned_state_dict)

            self._backup_params(model)
            with _torch.no_grad():
                if base_state_dict is not None:
                    base_sd = {k: v.detach().clone() if hasattr(v, "detach") else v for k, v in base_state_dict.items()}
                else:
                    base_sd = {n: p.data.clone() for n, p in model.named_parameters()}
                for name, param in model.named_parameters():
                    if name not in aligned_sd:
                        continue
                    base_w = base_sd[name].to(param.device)
                    aligned_w = aligned_sd[name].to(param.device)
                    delta = aligned_w - base_w
                    if max_delta_norm is not None:
                        norm = delta.norm()
                        if norm > float(max_delta_norm):
                            delta = delta * (float(max_delta_norm) / norm.clamp(min=1e-12))
                    param.data.copy_(base_w + alpha * delta)
            return

        # PEFT adapter merge path.
        try:
            from peft import PeftModel  # type: ignore[import]
        except ImportError:
            raise RuntimeError(
                "peft is required for SafeLoRAPatch.apply_to_model() with adapter paths. "
                "Install peft or provide aligned_state_dict_path instead."
            )

        base_adapter_path = self.params.get("base_adapter_path")
        aligned_adapter_path = self.params.get("aligned_adapter_path")
        if not aligned_adapter_path:
            raise ValueError(
                "SafeLoRAPatch.apply_to_model() requires 'aligned_adapter_path' "
                "or 'aligned_state_dict_path' in params."
            )

        self._backup_params(model)
        # Load and merge aligned adapter.
        peft_model = PeftModel.from_pretrained(model, aligned_adapter_path)
        with _torch.no_grad():
            for name, param in model.named_parameters():
                # Find the aligned param in the peft model.
                aligned_param = dict(peft_model.named_parameters()).get(name)
                if aligned_param is None:
                    continue
                base_w = self._model_param_backup[name].to(param.device)  # type: ignore[index]
                aligned_w = aligned_param.data.to(param.device)
                delta = aligned_w - base_w
                if max_delta_norm is not None:
                    norm = delta.norm()
                    if norm > float(max_delta_norm):
                        delta = delta * (float(max_delta_norm) / norm.clamp(min=1e-12))
                param.data.copy_(base_w + alpha * delta)
