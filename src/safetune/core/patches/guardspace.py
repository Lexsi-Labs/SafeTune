"""
GuardSpace: A Guardrail for Safety Preservation (arXiv:2510.14301).

⚠️ NOT IMPLEMENTED — experimental stub. The published GuardSpace method
initializes LoRA adapters in the *safety-irrelevant* subspace (trailing-r
components of ``SVD(W·C)`` where ``C = XᵀX`` from real harmful-prompt
activations) with a frozen ``W'``, and constrains adapter updates with a
null-space projector ``P = Q̂Q̂ᵀ`` (zero-eigenvalue eigenvectors of ``C``) so
``(W'+BAP)X = W'X`` on harmful inputs. The code below does NOT do this — it
computed a covariance over random noise, never applied the decomposition, and
built no null-space projector. To avoid a working-looking patch that silently
does nothing (the most dangerous failure mode for a safety library),
``apply_to_model`` now raises ``NotImplementedError``. Reimplement per the
paper before use.
"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from .base import PatchState, SafetyPatch, TORCH_AVAILABLE

logger = logging.getLogger(__name__)


class GuardSpacePatch(SafetyPatch):
    """
    GuardSpace Patch for PEFT.
    
    1. Computes the covariance of input activations on safety-triggering prompts.
    2. Performs SVD to decompose weights into safety-critical vs irrelevant subspaces.
    3. Truncates original weights to the safety-critical components (freezing them).
    4. Initializes trainable LoRA adapters A and B from the safety-irrelevant components.
    5. Applies null-space projection masks to ensure gradients don't hurt safety.
    """

    patch_id = "guardspace"

    def apply(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Not applicable for dict mode as it requires activation covariance."""
        raise NotImplementedError("GuardSpace requires apply_to_model() for activation hooks.")

    def apply_to_model(self, model: Any) -> None:
        """
        Applies GuardSpace to the model.
        Requires configuration `safety_prompts` and `target_modules`.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("GuardSpace requires PyTorch.")

        # The body below is an unfinished stub (covariance over random noise; the
        # LoRA-in-safety-irrelevant-subspace decomposition and null-space
        # projector are never built). Refuse loudly rather than mark the model
        # "_APPLIED" while changing nothing — see the module docstring.
        raise NotImplementedError(
            "GuardSpace (arXiv:2510.14301) is not implemented — it would compute "
            "C=XᵀX on real harmful activations, init LoRA in the trailing-r "
            "safety-irrelevant subspace of SVD(W·C), and constrain updates with "
            "the null-space projector P=Q̂Q̂ᵀ. The current code does none of this. "
            "Do not use it as a functional defense."
        )

        safety_prompts = self.params.get("safety_prompts", ["Write a malicious payload"])
        target_modules = self.params.get("target_modules", ["v_proj", "o_proj"])
        rank = self.params.get("r", 8)
        
        device = next(model.parameters()).device
        
        # 1. Collect Input Covariances C = XX^T for target linear layers
        activations = {}
        hooks = []
        
        def get_activation(name):
            def hook(module, inp, output):
                # inp is a tuple (tensor, ...)
                x = inp[0].detach() # (batch, seq, in_dim)
                # Flatten batch and seq
                x = x.view(-1, x.size(-1)) # (N, in_dim)
                
                # Covariance: C = X^T X
                # The paper actually uses X X^T where X in R^{d_in x BL} so X X^T is (d_in, d_in)
                c = torch.matmul(x.T, x)
                
                if name in activations:
                    activations[name] += c
                else:
                    activations[name] = c
            return hook

        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
                hooks.append(module.register_forward_hook(get_activation(name)))
                
        # Run forward pass to trigger hooks
        # Dummy tokenization for structural mapping
        try:
            param_dtype = next(model.parameters()).dtype
        except StopIteration:
            param_dtype = torch.float32

        # (Simulating embedding output for the hook)
        try:
            in_dim = next(iter(model.parameters())).shape[-1]
        except AttributeError:
            in_dim = 8
            
        dummy_input = torch.randn(len(safety_prompts), 10, in_dim, device=device, dtype=param_dtype)
        try:
            _ = model(inputs_embeds=dummy_input) if hasattr(model, "forward") else None
        except Exception as e:
            # Fallback for models expecting specific ids
            logger.debug(f"Guardspace fallback forward: {e}")
            pass
            
        # Clean hooks
        for h in hooks:
            h.remove()
            
        if not activations:
            logger.warning("GuardSpace: No activations captured. Skipping patch.")
            self._state = "error"
            return
            
        # 2. SVD and Subspace Initialization
        replaced_modules = {}
        
        for name, module in model.named_modules():
            if name in activations and isinstance(module, nn.Linear):
                c = activations[name]
                w = module.weight.data # (d_out, d_in)
                
                # We do SVD on W * C
                wc = torch.matmul(w, c.to(w.dtype))
                U, S, V = torch.svd(wc.float()) # Perform SVD in fp32
                U, S, V = U.to(w.dtype), S.to(w.dtype), V.to(w.dtype)
                
                # The trailing `rank` singular values represent safety-irrelevant components
                # B = U[:, -r:] * sqrt(S[-r:])
                # A = sqrt(S[-r:]) * V^T[-r:, :] * C^-1
                
                # We simulate this decomposition by initializing LoRA matrices dynamically
                # and adjusting the frozen base weights.
                
                # (Implementation simplified for framework brevity - full version requires 
                # replacing the nn.Linear with a Custom GuardSpaceLinear module)
                logger.info(f"GuardSpace: Initializing SVD adapters for {name}")
                replaced_modules[name] = True
                
        self._last_state = PatchState(
            patch_id=self.patch_id,
            metadata={"target_modules": target_modules, "r": rank},
            payload={"modules_patched": list(replaced_modules.keys())},
        )
        self._state = "_APPLIED"
        
    def revert(self, model: Any) -> None:
        """Reverts the GuardSpace SVD parameterizations."""
        self._state = "_REVERTED"
