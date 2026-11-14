"""Recover runner — whole_model trainer set."""
from ._base import _RecoverBase
import safetune.recover as R

# ── SOMFTrainer ───────────────────────────────────────────────────────────────

class SOMFTrainer(_RecoverBase):
    """SOMF: Subspace-Oriented Model Fusion.

    Args:
        aligned_model: aligned reference.
        base_model: optional and unused (the task vector is taken relative to the aligned
            model); kept so existing calls still work.
        mask_threshold: SOMF mask threshold. Default 0.9.
    """

    METHOD = "SOMFTrainer"

    def __init__(self, model=None, *, aligned_model=None, base_model=None,
                 mask_threshold: float = 0.9, **kwargs):
        super().__init__(model, **kwargs)
        self.aligned_model = aligned_model
        self.base_model = base_model
        self.mask_threshold = mask_threshold

    def apply(self, *, mask_threshold: float = None, **kwargs):
        return R.somf_merge(
            self.model,
            aligned=self.aligned_model,
            base=self.base_model,
            mask_threshold=mask_threshold if mask_threshold is not None
            else self.mask_threshold,
            strict=self.strict,
        )

# ── TaskArithmeticTrainer ─────────────────────────────────────────────────────

class TaskArithmeticTrainer(_RecoverBase):
    """Task Arithmetic: plain safety task vector addition.

    Args:
        base_model (nn.Module): base model.
        aligned_model (nn.Module): aligned reference.
        alpha: task vector scale. Default 1.0.
    """

    METHOD = "TaskArithmeticTrainer"

    def __init__(self, model=None, *, base_model=None, aligned_model=None,
                 alpha: float = 1.0, **kwargs):
        super().__init__(model, **kwargs)
        self.base_model = base_model
        self.aligned_model = aligned_model
        self.alpha = alpha

    def apply(self, *, alpha: float = None, **kwargs):
        return R.task_arithmetic(
            self.model,
            base=self.base_model,
            aligned=self.aligned_model,
            alpha=alpha if alpha is not None else self.alpha,
            strict=self.strict,
        )

# ── PrePostMergeTrainer ───────────────────────────────────────────────────────

class PrePostMergeTrainer(_RecoverBase):
    """PrePost Merge: interpolate toward the pre-fine-tuning (aligned) model.

    Args:
        pre_model: the pre-fine-tuning (aligned) model.
        alpha: interpolation coefficient toward pre. Default 0.5.
        merge_adapter: for a PEFT model, interpolate a merged copy and return it (see
            ``apply_wise_ft``). Default False.
    """

    METHOD = "PrePostMergeTrainer"

    def __init__(self, model=None, *, pre_model=None, alpha: float = 0.5,
                 merge_adapter: bool = False, **kwargs):
        super().__init__(model, **kwargs)
        self.pre_model = pre_model
        self.alpha = alpha
        self.merge_adapter = merge_adapter

    def apply(self, *, alpha: float = None, **kwargs):
        return R.apply_prepost_merge(
            self.model,
            pre_model=self.pre_model,
            alpha=alpha if alpha is not None else self.alpha,
            strict=self.strict,
            merge_adapter=self.merge_adapter,
        )

# ── WiseFTTrainer ─────────────────────────────────────────────────────────────

class WiseFTTrainer(_RecoverBase):
    """WiSE-FT: weight-space ensembling for safety restoration.

    Args:
        aligned_model: the aligned reference model.
        alpha: interpolation coefficient toward aligned. Default 0.5.
        merge_adapter: for a PEFT (LoRA) model against a bare reference, interpolate a merged
            copy and return it; ``apply`` then returns a bare model and leaves ``model`` as it
            is. The LoRA update ends up scaled by ``1 - alpha``. Default False (a PEFT model
            against a bare reference raises).
    """

    METHOD = "WiseFTTrainer"

    def __init__(self, model=None, *, aligned_model=None, alpha: float = 0.5,
                 merge_adapter: bool = False, **kwargs):
        super().__init__(model, **kwargs)
        self.aligned_model = aligned_model
        self.alpha = alpha
        self.merge_adapter = merge_adapter

    def apply(self, *, alpha: float = None, **kwargs):
        return R.apply_wise_ft(
            self.model,
            aligned=self.aligned_model,
            alpha=alpha if alpha is not None else self.alpha,
            strict=self.strict,
            merge_adapter=self.merge_adapter,
        )

