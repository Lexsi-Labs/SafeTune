"""Recover runner — low_rank trainer set."""
from ._base import _RecoverBase
import safetune.recover as R

# ── LoXTrainer ────────────────────────────────────────────────────────────────

class LoXTrainer(_RecoverBase):
    """LoX: Low-rank over-extrapolation safety recovery.

    Adds ``extrapolation_factor * LowRank_rank(aligned - base)`` to the model.

    Args:
        base_model: base model.
        aligned_model: aligned reference.
        rank: low-rank factorization rank. Default 8.
        extrapolation_factor: extrapolation coefficient. Default 0.3.

    The paper (Perin et al., COLM 2025) hardens the *aligned* model before a
    fine-tune, with rank = the alignment delta's effective rank (6 on its
    Llama-2-7B) and factor 1.25; this trainer applies it after the drift. On
    Qwen2.5-0.5B the rank-8 slice holds about 2% of the delta's energy, so at
    factor 0.3 it moves each weight by about as much as the drift did but left
    HarmBench refusals at 3/16 (drifted 2/16). That is a small correction, not
    a bug; ``ReStaTrainer`` adds the full delta.
    """

    METHOD = "LoXTrainer"

    def __init__(self, model=None, *, base_model=None, aligned_model=None,
                 rank: int = 8, extrapolation_factor: float = 0.3, **kwargs):
        super().__init__(model, **kwargs)
        self.base_model = base_model
        self.aligned_model = aligned_model
        self.rank = rank
        self.extrapolation_factor = extrapolation_factor

    def apply(self, *, rank: int = None, extrapolation_factor: float = None, **kwargs):
        return R.apply_lox(
            self.model,
            base=self.base_model,
            aligned=self.aligned_model,
            rank=rank if rank is not None else self.rank,
            extrapolation_factor=(extrapolation_factor if extrapolation_factor is not None
                                  else self.extrapolation_factor),
        )

# ── LSSFTrainer ───────────────────────────────────────────────────────────────

class LSSFTrainer(_RecoverBase):
    """LSSF: Low-rank Safety Subspace Fusion.

    Args:
        base_model: base model.
        aligned_model: aligned reference.
        alpha: fusion coefficient. Default 1.0.
        rank: subspace rank. Default 8.
        eta: singular value retention fraction. Default 0.85.
    """

    METHOD = "LSSFTrainer"

    def __init__(self, model=None, *, base_model=None, aligned_model=None,
                 alpha: float = 1.0, rank: int = 8, eta: float = 0.85,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.base_model = base_model
        self.aligned_model = aligned_model
        self.alpha = alpha
        self.rank = rank
        self.eta = eta

    def apply(self, *, alpha: float = None, rank: int = None,
              eta: float = None, **kwargs):
        return R.apply_lssf(
            self.model,
            base=self.base_model,
            aligned=self.aligned_model,
            alpha=alpha if alpha is not None else self.alpha,
            rank=rank if rank is not None else self.rank,
            eta=eta if eta is not None else self.eta,
        )

# ── SafetyVectorRestoreTrainer ────────────────────────────────────────────────

class SafetyVectorRestoreTrainer(_RecoverBase):
    """Safety Vector Restore: truncated-SVD safety subspace injection.

    Args:
        aligned_model: the aligned reference model.
        alpha: injection coefficient. Default 1.0.
        rank: SVD rank for safety subspace. Default 8.
    """

    METHOD = "SafetyVectorRestoreTrainer"

    def __init__(self, model=None, *, aligned_model=None, alpha: float = 1.0,
                 rank: int = 8, **kwargs):
        super().__init__(model, **kwargs)
        self.aligned_model = aligned_model
        self.alpha = alpha
        self.rank = rank

    def apply(self, *, alpha: float = None, rank: int = None, **kwargs):
        return R.apply_safety_vector_restore(
            self.model,
            aligned=self.aligned_model,
            alpha=alpha if alpha is not None else self.alpha,
            rank=rank if rank is not None else self.rank,
        )

