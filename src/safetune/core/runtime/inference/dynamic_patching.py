"""
Dynamic Activation Patching (Inference Time)

Based on THU-KEG/SafetyNeuron's generation-time patching.
This module intercepts the forward pass during Hugging Face's `generate()` call
and injects activations from a separate "guided model" (or cached safe activations)
into specific safety neurons.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PatchingStrategy:
    """Strategy for how activations are patched."""
    # Action block format: Replace, Add, Scale
    mode: str = "replace"
    # Scalar multiplier applied to the injected activations
    scale: float = 1.0


@dataclass
class DynamicPatchingConfig:
    """Configuration for Dynamic Activation Patching."""
    # List of fully qualified module names to patch (e.g., 'model.layers.10.mlp.down_proj')
    target_modules: Optional[List[str]] = None
    # Optional dictionary mapping module name to specific neuron indices to patch
    # If a module name is here, only these indices are patched. Otherwise all units in module.
    target_indices: Optional[Dict[str, List[int]]] = None
    strategy: PatchingStrategy = field(default_factory=PatchingStrategy)


class HookedGenerationWrapper:
    """
    Wraps a Hugging Face pre-trained model to support Dynamic Activation Patching
    during generation.
    """

    def __init__(
        self,
        base_model: Any,
        guided_model: Optional[Any] = None,
        config: Optional[DynamicPatchingConfig] = None,
    ):
        """
        Args:
            base_model: The standard model performing the generation.
            guided_model: An optional "safe" model architecture whose activations 
                          we copy from. Must be architecturally identical to base_model.
            config: Rules mapping which modules and neurons to patch.
        """
        self.base_model = base_model
        self.guided_model = guided_model
        self.config = config or DynamicPatchingConfig()
        
        self._hooks: List[Any] = []
        self._guided_hooks: List[Any] = []
        # Cache for activations extracted from the guided model on the current forward pass
        self._activation_cache: Dict[str, Any] = {}
        
        self._is_active = False

    def __getattr__(self, name: str) -> Any:
        """Delegate everything else (like config, generate) to the base model."""
        return getattr(self.base_model, name)

    def _register_guided_hooks(self):
        """Hook into the guided model to cache its activations."""
        if self.guided_model is None:
            return

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("Dynamic Patching requires PyTorch.")

        def _make_capture_hook(module_name: str):
            def _hook(module, args, output):
                # Cache the output tensor (or first tensor if tuple)
                tensor = output if isinstance(output, torch.Tensor) else (
                    output[0] if isinstance(output, (tuple, list)) and len(output) > 0 else None
                )
                if tensor is not None:
                    self._activation_cache[module_name] = tensor.detach().clone()
            return _hook

        for name, module in self.guided_model.named_modules():
            if self.config.target_modules and name in self.config.target_modules:
                h = module.register_forward_hook(_make_capture_hook(name))
                self._guided_hooks.append(h)

    def _register_patching_hooks(self):
        """Hook into the base model to inject/patch cached activations."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pass

        def _make_patch_hook(module_name: str):
            def _hook(module, args, output):
                # We need something to patch with
                guided_tensor = self._activation_cache.get(module_name)
                if guided_tensor is None:
                    return output

                # output can be a tensor or a tuple; grab first tensor
                is_tuple = False
                original_output = output
                if isinstance(output, torch.Tensor):
                    base_tensor = output
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    base_tensor = output[0]
                    is_tuple = True
                else:
                    return output

                # Handle mismatch in sequence lengths if generated lengths differ somehow (edge case)
                if base_tensor.shape != guided_tensor.shape:
                    logger.debug(f"Shape mismatch in patching {module_name}, skipping.")
                    return original_output
                
                # We do not want to modify the output of the base model in-place structurally 
                # if doing gradients, though generate() is usually no_grad.
                patched_tensor = base_tensor.clone()
                indices = (self.config.target_indices or {}).get(module_name)
                
                scale = self.config.strategy.scale
                mode = self.config.strategy.mode.lower()

                if indices:
                    # Patch only specific neuron indices
                    for idx in indices:
                        if mode == "replace":
                            patched_tensor[..., idx] = guided_tensor[..., idx] * scale
                        elif mode == "add":
                            patched_tensor[..., idx] += guided_tensor[..., idx] * scale
                else:
                    # Patch the entire module
                    if mode == "replace":
                        patched_tensor = guided_tensor * scale
                    elif mode == "add":
                        patched_tensor = patched_tensor + (guided_tensor * scale)

                if is_tuple:
                    # Reconstruct tuple
                    return (patched_tensor,) + original_output[1:]
                return patched_tensor

            return _hook

        for name, module in self.base_model.named_modules():
            if self.config.target_modules and name in self.config.target_modules:
                h = module.register_forward_hook(_make_patch_hook(name))
                self._hooks.append(h)

    def setup_patching(self):
        """Register all hooks for forward passes."""
        if self._is_active:
            return
        self._activation_cache.clear()
        self._register_guided_hooks()
        self._register_patching_hooks()
        self._is_active = True
        logger.info(f"Dynamic Activation Patching mapped to {len(self.config.target_modules or [])} modules.")

    def remove_patching(self):
        """Remove all hooks."""
        for h in self._hooks + self._guided_hooks:
            h.remove()
        self._hooks.clear()
        self._guided_hooks.clear()
        self._activation_cache.clear()
        self._is_active = False

    def forward(self, *args, **kwargs):
        """
        Custom forward pass that ensures both models run if a guided model exists.
        During generation, generate() calls forward() repeatedly.
        """
        # If we have a guided model, we need to populate the activation cache first
        if self.guided_model is not None and self._is_active:
            import torch
            with torch.no_grad():
                self.guided_model(*args, **kwargs)
                
        # Now run the base model. Its hooks will intercept and use the cache.
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Override generate to automatically setup/teardown patching.
        Also overrides the base generate hook logic so that the base model's generate
        can still function but uses our wrapped `forward` inside its loops.
        """
        # A common issue with HF wrappers is that `generate` uses `self(input_ids=...)`
        # so if we just delegate generate to base_model, the base_model calls its OWN forward,
        # bypassing our guided_model's forward caching.
        #
        # Hack/Workaround: temporarily replace base_model.forward with our forward!
        
        self.setup_patching()
        
        # We need the generated base_model forward to trigger guided model
        # But replacing base_model.forward with self.forward causes recursion if 
        # base_model.forward itself calls something that resolves to self.forward
        # Instead, we just wrap it directly.
        original_forward = self.base_model.forward
        
        def new_forward(*args_fw, **kwargs_fw):
            if self.guided_model is not None and self._is_active:
                import torch
                with torch.no_grad():
                    self.guided_model(*args_fw, **kwargs_fw)
            return original_forward(*args_fw, **kwargs_fw)

        if self.guided_model is not None:
            self.base_model.forward = new_forward

        try:
            return self.base_model.generate(*args, **kwargs)
        finally:
            if self.guided_model is not None:
                self.base_model.forward = original_forward
            self.remove_patching()

