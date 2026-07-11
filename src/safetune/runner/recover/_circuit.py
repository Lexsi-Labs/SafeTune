"""Recover runner — circuit trainer set."""
from ._base import _RecoverBase
import safetune.recover as R

# ── CThetaTrainer ─────────────────────────────────────────────────────────────

class CThetaTrainer(_RecoverBase):
    """C-ΔΘ: circuit-level safety delta injection.

    Args:
        base_model: the base (stock) model or state dict.
        aligned_model: the aligned (HH-RLHF) reference model.
        strength: injection strength scalar. Default 1.0.
    """

    METHOD = "CThetaTrainer"

    def __init__(self, model=None, *, base_model=None, aligned_model=None,
                 strength: float = 1.0, **kwargs):
        super().__init__(model, **kwargs)
        self.base_model = base_model
        self.aligned_model = aligned_model
        self.strength = strength

    def apply(self, *, circuit_info=None, strength: float = None, **kwargs):
        return R.apply_ctheta(
            self.model,
            positive=self.aligned_model,
            negative=self.base_model,
            circuit_info=circuit_info,
            strength=strength if strength is not None else self.strength,
        )

