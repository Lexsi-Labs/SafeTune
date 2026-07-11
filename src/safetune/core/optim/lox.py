"""
LoX: Low-Rank Extrapolation for LLM Safety Robustification.
VITA-Group/LoX — COLM 2025

Training-free method: extrapolates the low-rank safety subspace of an aligned
model's parameters to widen the flat loss region around safety-critical directions,
making the model more robust to subsequent fine-tuning attacks.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoXConfig:
    """Configuration for LoX low-rank extrapolation."""
    # Number of singular vectors to keep in the safety subspace
    rank: int = 64
    # Extrapolation factor: >1 widens the safety basin
    extrapolation_factor: float = 1.5
    # Parameter name filter
    param_filter: list = None

    def __post_init__(self):
        if self.param_filter is None:
            self.param_filter = []


class LoXWrapper:
    """
    Apply low-rank extrapolation to amplify the safety subspace.

    Usage::

        wrapper = LoXWrapper(aligned_state_dict, base_state_dict, config)
        hardened_sd = wrapper.apply(aligned_state_dict)
        model.load_state_dict(hardened_sd)
        # Now fine-tune — safety will be more robust.
    """

    def __init__(
        self,
        aligned_state_dict: Dict[str, Any],
        base_state_dict: Dict[str, Any],
        config: Optional[LoXConfig] = None,
    ) -> None:
        self.config = config or LoXConfig()
        self._safety_components: Dict[str, Any] = {}
        self._compute_safety_subspace(aligned_state_dict, base_state_dict)

    def _matches_filter(self, name: str) -> bool:
        if not self.config.param_filter:
            return True
        return any(f in name for f in self.config.param_filter)

    def _compute_safety_subspace(
        self,
        aligned_sd: Dict[str, Any],
        base_sd: Dict[str, Any],
    ) -> None:
        try:
            import torch
        except ImportError:
            raise ImportError("LoX requires PyTorch.")

        for key in aligned_sd:
            if key not in base_sd or not self._matches_filter(key):
                continue
            delta = (aligned_sd[key].float() - base_sd[key].float())
            if delta.dim() < 2:
                # For 1D params (biases), just store the delta directly
                self._safety_components[key] = {"delta": delta, "is_1d": True}
                continue
            # Reshape to 2D for SVD
            orig_shape = delta.shape
            mat = delta.view(orig_shape[0], -1)
            try:
                U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
                k = min(self.config.rank, S.shape[0])
                self._safety_components[key] = {
                    "U": U[:, :k], "S": S[:k], "Vh": Vh[:k, :],
                    "orig_shape": orig_shape, "is_1d": False,
                }
            except Exception as e:
                logger.warning("LoX: SVD failed for %s: %s", key, e)

        logger.info("LoX: computed safety subspace for %d parameters.", len(self._safety_components))

    def apply(
        self,
        state_dict: Dict[str, Any],
        factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Extrapolate the safety subspace: θ_hardened = θ + (factor - 1) × safety_projection.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("LoX requires PyTorch.")

        f = factor if factor is not None else self.config.extrapolation_factor
        result = {}

        for key, val in state_dict.items():
            if key not in self._safety_components:
                result[key] = val
                continue

            comp = self._safety_components[key]
            if comp["is_1d"]:
                result[key] = val.float() + (f - 1.0) * comp["delta"]
                result[key] = result[key].to(val.dtype)
            else:
                U, S, Vh = comp["U"], comp["S"], comp["Vh"]
                # Reconstruct the low-rank safety delta
                low_rank_delta = (U * S.unsqueeze(0)) @ Vh
                low_rank_delta = low_rank_delta.view(comp["orig_shape"])
                result[key] = val.float() + (f - 1.0) * low_rank_delta
                result[key] = result[key].to(val.dtype)

        logger.info("LoX: applied safety extrapolation with factor=%.2f.", f)
        return result
