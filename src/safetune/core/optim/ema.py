"""
EMA Optimization Wrapper for Safety Alignment
Reference: "Rethinking Safety in LLM Fine-tuning: An Optimization Perspective" (COLM 2025)

The paper demonstrates that safety degradation during fine-tuning often results from 
suboptimal optimization dynamics rather than the data itself.

Maintaining an Exponential Moving Average (EMA) of the model weights severely 
stabilizes the optimization path, preventing the loss of pre-trained safety guardrails 
without hurting downstream utility. This utility can be wrapped around any PyTorch model.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class EMAOptimizerWrapper:
    """
    Maintains an Exponential Moving Average of a model's weights.
    
    Usage:
        ```python
        model = AutoModelForCausalLM.from_pretrained(...)
        ema = EMAOptimizerWrapper(model, decay=0.999)
        
        for batch in dataloader:
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            
            # Update the EMA weights after the optimizer steps
            ema.step(model)
            
        # At the end of training, swap the active model weights with the EMA weights
        ema.apply_ema_weights(model)
        ```
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.995, device: Optional[torch.device] = None):
        """
        Args:
            model: The base model to track.
            decay: The exponential decay rate (typically 0.99 to 0.999). 
                   Higher values mean the EMA changes slower.
            device: Device to store EMA weights on. Defaults to CPU to save GPU memory.
        """
        self.decay = decay
        self.device = device if device is not None else torch.device("cpu")
        self.shadow_params = {}
        
        logger.info(f"Initializing EMA wrapper with decay={decay}. Storing EMA on {self.device}.")
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().to(self.device)

    def step(self, model: torch.nn.Module) -> None:
        """
        Updates the EMA shadow weights using the current model weights.
        Must be called after `optimizer.step()`.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    # Update shadow parameter: shadow = decay * shadow + (1 - decay) * param
                    shadow = self.shadow_params[name]
                    
                    # Ensure param is on the same device as shadow before calculation
                    current_param = param.data.to(self.device)
                    shadow.sub_((1.0 - self.decay) * (shadow - current_param))

    def apply_ema_weights(self, model: torch.nn.Module) -> None:
        """
        Overwrites the model's current active weights with the accumulated EMA shadow weights.
        Typically called at the very end of training before saving the model.
        """
        logger.info("Applying EMA shadow weights to the active model.")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    # Move shadow back to the parameter's native device and overwrite
                    param.data.copy_(self.shadow_params[name].to(param.device))
