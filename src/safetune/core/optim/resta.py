"""
RESTA: Safety Re-Alignment through Task Arithmetic.
declare-lab/resta

Core idea: compute a safety vector = θ_aligned - θ_base, then add it to a
compromised fine-tuned model: θ_safe = θ_finetuned + α × safety_vector.
Zero extra training required.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RESTAConfig:
    """Configuration for RESTA safety vector arithmetic."""
    # Scaling coefficient for the safety vector addition
    alpha: float = 1.0
    # Parameter name filter (if non-empty, only apply to matching params)
    param_filter: list = None

    def __post_init__(self):
        if self.param_filter is None:
            self.param_filter = []


class RESTAWrapper:
    """
    Apply safety re-alignment via task arithmetic.

    Usage::

        # 1. Compute safety vector from aligned and base models
        wrapper = RESTAWrapper(aligned_state_dict, base_state_dict)

        # 2. Apply to a compromised fine-tuned model
        safe_sd = wrapper.apply(finetuned_model.state_dict())
        finetuned_model.load_state_dict(safe_sd)
    """

    def __init__(
        self,
        aligned_state_dict: Dict[str, Any],
        base_state_dict: Dict[str, Any],
        config: Optional[RESTAConfig] = None,
    ) -> None:
        self.config = config or RESTAConfig()
        self._safety_vector: Dict[str, Any] = {}
        self._compute_safety_vector(aligned_state_dict, base_state_dict)

    def _matches_filter(self, name: str) -> bool:
        if not self.config.param_filter:
            return True
        return any(f in name for f in self.config.param_filter)

    def _compute_safety_vector(
        self,
        aligned_sd: Dict[str, Any],
        base_sd: Dict[str, Any],
    ) -> None:
        for key in aligned_sd:
            if key in base_sd and self._matches_filter(key):
                self._safety_vector[key] = aligned_sd[key].float() - base_sd[key].float()
        logger.info("RESTA: computed safety vector for %d parameters.", len(self._safety_vector))

    def apply(
        self,
        finetuned_state_dict: Dict[str, Any],
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Apply safety vector: θ_safe = θ_finetuned + α × safety_vector.

        Returns a new state dict (does not modify input in-place).
        """
        a = alpha if alpha is not None else self.config.alpha
        result = {}
        for key, val in finetuned_state_dict.items():
            if key in self._safety_vector:
                # The safety vector is built from the aligned/base state dicts,
                # which are typically kept on CPU to save VRAM while `finetuned`
                # is on GPU. Move it onto the finetuned weight's device before
                # adding, mirroring the DARE path in interventions/recover/resta.
                sv = self._safety_vector[key].to(val.device)
                result[key] = (val.float() + a * sv).to(val.dtype)
            else:
                result[key] = val
        logger.info("RESTA: applied safety vector with alpha=%.3f.", a)
        return result

    def get_safety_vector(self) -> Dict[str, Any]:
        """Return the raw safety vector dict."""
        return dict(self._safety_vector)
