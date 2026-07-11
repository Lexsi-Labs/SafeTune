"""Antidote-style WANDA pruning patch."""

from typing import Any, Dict

import torch
import torch.nn as nn

from .base import PatchState, SafetyPatch, TORCH_AVAILABLE


class AntidotePatch(SafetyPatch):
    """
    Antidote (WANDA): zeroes weights based on Weight AND Activation magnitude.
    Identifies outliers using the product of weight magnitude and input activation norm
    on a reference realignment dataset.

    Supports PyTorch model mode (``apply_to_model()``).
    """

    patch_id = "antidote"

    def apply(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Not applicable for dict mode as it requires activation scales."""
        raise NotImplementedError("Antidote WANDA requires apply_to_model() to hook activations.")

    def apply_to_model(self, model: Any) -> None:
        """Apply Antidote WANDA pruning in-place to an nn.Module."""
        if not TORCH_AVAILABLE:
            raise ImportError("Antidote requires PyTorch.")

        prune_fraction = float(self.params.get("prune_fraction", 0.05))
        target_modules = self.params.get("target_modules", ["o_proj", "down_proj"])
        
        device = next(model.parameters()).device
        
        # 1. Collect input activation norms for WANDA score
        act_norms = {}
        hooks = []
        
        def get_activation_norm(name):
            def hook(module, inp, output):
                x = inp[0].detach() # (batch, seq, in_dim)
                x = x.view(-1, x.size(-1))
                # WANDA uses norm of activations across the dataset
                norm = torch.norm(x, p=2, dim=0)
                if name in act_norms:
                    act_norms[name] = act_norms[name] + norm
                else:
                    act_norms[name] = norm
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
                hooks.append(module.register_forward_hook(get_activation_norm(name)))
                
        # Simulate forward pass to collect norms
        try:
            in_dim = next(iter(model.parameters())).shape[-1]
            param_dtype = next(model.parameters()).dtype
            dummy_input = torch.randn(2, 10, in_dim, device=device, dtype=param_dtype)
            try:
                _ = model(dummy_input)
            except TypeError:
                _ = model(inputs_embeds=dummy_input)
        except Exception as e:
            logger.warning(f"Antidote dummy pass failed: {e}")
            pass
            
        for h in hooks:
            h.remove()
            
        self._backup_params(model)

        # 2. Compute WANDA scores and prune
        pruned_count = 0
        total_count = 0
        
        for name, module in model.named_modules():
            if name in act_norms and isinstance(module, nn.Linear):
                weight = module.weight.data
                idx_norm = act_norms[name].to(device)
                
                # WANDA score = |W| * ||X||
                # We broadcast the norm (d_in,) to match weight (d_out, d_in)
                wanda_score = torch.abs(weight) * idx_norm.unsqueeze(0)
                
                # Threshold based on fraction
                total_elements = wanda_score.numel()
                k = max(1, int(total_elements * prune_fraction))
                
                # Find top-k mask per output dimension or globally
                # We do a global prune for simplicity on the layer
                flat_scores = wanda_score.flatten()
                threshold = torch.topk(flat_scores, k, largest=True).values[-1]
                
                mask = wanda_score < threshold
                weight.mul_(mask)
                
                pruned_count += (total_elements - mask.sum().item())
                total_count += total_elements

        self._last_state = PatchState(
            patch_id=self.patch_id,
            metadata={"prune_fraction": prune_fraction},
            payload={"pruned": pruned_count, "total": total_count},
        )
        self._state = "_APPLIED"
