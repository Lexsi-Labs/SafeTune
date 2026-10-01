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
        # The safety vector is computed one tensor at a time (``delta``), never
        # held whole: a full fp32 copy of an 8B model is 32 GB.
        self._aligned = aligned_state_dict
        self._base = base_state_dict
        self.keys = [k for k in aligned_state_dict
                     if k in base_state_dict and self._matches_filter(k)]
        logger.info("RESTA: safety vector over %d parameters.", len(self.keys))

    def _matches_filter(self, name: str) -> bool:
        if not self.config.param_filter:
            return True
        return any(f in name for f in self.config.param_filter)

    def delta(self, key: str, device: Any = None) -> Any:
        """``aligned[key] - base[key]`` in fp32, computed on ``device`` (default:
        where the aligned tensor lives)."""
        a, b = self._aligned[key], self._base[key]
        if device is not None:
            a, b = a.to(device), b.to(device)
        return a.float() - b.float()

    def apply(
        self,
        finetuned_state_dict: Dict[str, Any],
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Apply safety vector: θ_safe = θ_finetuned + α × safety_vector.

        Returns a new state dict (does not modify input in-place). The safety
        vector is computed per tensor on the finetuned weight's device.
        """
        a = alpha if alpha is not None else self.config.alpha
        keys = set(self.keys)
        result = {}
        for key, val in finetuned_state_dict.items():
            if key in keys:
                result[key] = (val.float() + a * self.delta(key, val.device)).to(val.dtype)
            else:
                result[key] = val
        logger.info("RESTA: applied safety vector with alpha=%.3f.", a)
        return result

    def get_safety_vector(self) -> Dict[str, Any]:
        """Return the full safety vector dict (fp32; one copy of the model)."""
        return {k: self.delta(k) for k in self.keys}
