"""
Safety-Aware Probing Optimization (SAP).
Reference: "Mitigating Fine-tuning Risks in LLMs via Safety-Aware Probing Optimization"
Source: github.com/ChengcanWu/SAP

This module implements SAP as a standalone context manager that shifts the 
model weights in the direction of a safety contrastive gradient before computing 
the downstream user loss, and subsequently removes the shift before the 
final optimizer step. 
"""

import logging
from contextlib import contextmanager
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class SafetyAwareProbingWrapper:
    """
    Implements the Safety-Aware Probing (SAP) algorithm.
    
    SAP modifies the standard fine-tuning step by first shifting the model weights 
    along a safety gradient vector (derived via contrastive loss on safety data), 
    computing the user utility loss on those "safe" weights, and then shifting 
    the weights back before the final backpropagation update.
    """
    
    def __init__(self, model: nn.Module, grad_rate: float = 0.1):
        """
        Args:
            model: The PyTorch model to optimize.
            grad_rate: The scaling factor (alpha) to govern the size of the 
                       safety weight perturbation.
        """
        self.model = model
        self.grad_rate = grad_rate

    @staticmethod
    def compute_contrastive_safety_gradient(
        model: nn.Module, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        chosen_labels: torch.Tensor, 
        rejected_labels: torch.Tensor,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the contrastive gradient direction that pushes the model 
        towards safe behavior and away from harmful behavior.
        
        This assumes the user runs this in a `with torch.enable_grad():` block
        and manages zeroing gradients before and after this call if necessary.

        The returned gradient dict is globally L2-normalized (a unit
        direction), so downstream weight shifts scale purely as
        ``grad_rate * ||W||`` independent of the raw gradient magnitude.
        """
        import torch.nn.functional as F

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Causal LM shift: logits at position t predict the token at t+1.
        # ``chosen_labels`` / ``rejected_labels`` are the raw token ids of the
        # full (prompt + completion) sequence, position-aligned with
        # ``input_ids`` (see ``sap_contrastive_dataset``), so the standard
        # shift applies: compare logits[..., :-1, :] with labels[..., 1:].
        shift_logits = logits[:, :-1, :]
        shift_chosen = chosen_labels[:, 1:]
        shift_rejected = rejected_labels[:, 1:]

        # log probs of chosen (safe) outputs
        chosen_logps = -F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_chosen.reshape(-1),
            reduction='none'
        ).view(shift_chosen.shape).sum(dim=1)

        # log probs of rejected (harmful) outputs
        rejected_logps = -F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_rejected.reshape(-1),
            reduction='none'
        ).view(shift_rejected.shape).sum(dim=1)

        # contrastive objective
        log_ratio = (chosen_logps - rejected_logps) / temperature
        loss = -F.logsigmoid(log_ratio).mean()
        
        # Compute gradient
        loss.backward()
        
        safety_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                # Store the exact gradient at this moment
                safety_grads[name] = param.grad.detach().clone()

        # Globally L2-normalize the harmful direction so that consumers
        # shifting weights by ``epsilon * g`` obtain a perturbation of
        # magnitude exactly ``epsilon = grad_rate * ||W||`` — matching the
        # ``merge_lora_parameters(model, normalize(g), grad_rate * norm)``
        # scaling of the authors' SAPcode/train.py.  Without this the shift
        # scales with the raw gradient norm (NaN-prone when large, vanishing
        # when small).
        if safety_grads:
            total_norm = torch.sqrt(
                sum(g.detach().float().pow(2).sum() for g in safety_grads.values())
            )
            if torch.isfinite(total_norm) and total_norm > 0:
                for name, g in safety_grads.items():
                    safety_grads[name] = g / total_norm.to(device=g.device, dtype=g.dtype)

        return safety_grads

    def _normalize_and_flatten(self, grads: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Size]]:
        flat_vector = torch.cat([g.flatten() for g in grads.values()])
        shapes = {k: v.shape for k, v in grads.items()}
        
        # Safe normalize
        eps = 1e-8
        norm = flat_vector.norm(p=2, dim=-1, keepdim=True)
        normalized_vector = flat_vector / (norm + eps)
        
        return normalized_vector, shapes

    def _unflatten_and_apply(
        self, 
        flat_vector: torch.Tensor, 
        shapes: Dict[str, torch.Size], 
        scale: float
    ):
        start = 0
        for name, param in self.model.named_parameters():
            if name in shapes:
                shape = shapes[name]
                numel = shape.numel()
                grad_slice = flat_vector[start : start + numel].view(shape)
                
                # Perturb the actual underlying weights
                param.data.copy_(param.data + scale * grad_slice)
                start += numel

    @contextmanager
    def probe_safe_parameters(self, safety_gradients: Dict[str, torch.Tensor]):
        """
        Context manager that shifts model weights into the "safe subspace" 
        defined by the `safety_gradients`, yields control to compute the 
        standard utility loss, and then perfectly reverses the shift.
        
        Usage:
            safety_grads = wrapper.compute_contrastive_safety_gradient(...)
            model.zero_grad()
            
            with wrapper.probe_safe_parameters(safety_grads):
                loss = compute_utility_loss(model, user_batch)
                loss.backward()
                
            optimizer.step()
        """
        if not safety_gradients:
            yield
            return
            
        flat_norm_grad, shapes = self._normalize_and_flatten(safety_gradients)
        
        # Calculate total parameter norm for scaling 
        # (similar to get_lora_params norm_only in original SAP code)
        total_param_norm = torch.cat([
            param.flatten() 
            for name, param in self.model.named_parameters() 
            if name in shapes
        ]).norm(p=2, dim=-1).item()
        
        shift_magnitude = self.grad_rate * total_param_norm
        
        # 1. Shift INTO the safety subspace
        with torch.no_grad():
            self._unflatten_and_apply(flat_norm_grad, shapes, scale=shift_magnitude)
            
        try:
            # 2. Yield to user to compute standard forward/backward passes mapping utility space
            yield
        finally:
            # 3. Shift OUT of the safety subspace perfectly 
            with torch.no_grad():
                self._unflatten_and_apply(flat_norm_grad, shapes, scale=-shift_magnitude)
