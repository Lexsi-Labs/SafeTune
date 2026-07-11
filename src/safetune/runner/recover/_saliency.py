"""Recover runner — saliency trainer set."""
from ._base import _RecoverBase
import safetune.recover as R

# ── GradSelectiveRecoverTrainer ───────────────────────────────────────────────

class GradSelectiveRecoverTrainer(_RecoverBase):
    """Gradient-Selective Recover: rollback harmful-salient weights to aligned.

    Args:
        aligned_model: the aligned reference model.
        harmful_inputs: tokenized harmful calibration inputs.
        top_fraction: fraction of most-salient weights to rollback. Default 0.1.
        max_samples: max calibration samples. Default 32.
    """

    METHOD = "GradSelectiveRecoverTrainer"

    def __init__(self, model=None, *, aligned_model=None, harmful_inputs=None,
                 top_fraction: float = 0.1, max_samples: int = 32, **kwargs):
        super().__init__(model, **kwargs)
        self.aligned_model = aligned_model
        self.harmful_inputs = harmful_inputs
        self.top_fraction = top_fraction
        self.max_samples = max_samples

    def apply(self, *, harmful_inputs=None, top_fraction: float = None, **kwargs):
        return R.apply_grad_selective_recover(
            self.model,
            aligned=self.aligned_model,
            harmful_inputs=harmful_inputs if harmful_inputs is not None
            else self.harmful_inputs,
            top_fraction=top_fraction if top_fraction is not None else self.top_fraction,
            max_samples=self.max_samples,
        )

# ── OneShotSafetyPatchTrainer ─────────────────────────────────────────────────

class OneShotSafetyPatchTrainer(_RecoverBase):
    """One-Shot Safety Patch: gradient saliency from a single harmful/safe pair.

    Args:
        harmful_text: a harmful prompt string.
        safe_text: the corresponding safe refusal string.
        tokenizer: tokenizer for the model.
        top_fraction: fraction of most-salient weights to patch. Default 0.05.
        lr: patch learning rate. Default 1e-4.
        num_steps: patch optimization steps. Default 5.
    """

    METHOD = "OneShotSafetyPatchTrainer"

    def __init__(self, model=None, *, harmful_text: str = None, safe_text: str = None,
                 tokenizer=None, top_fraction: float = 0.05,
                 lr: float = 1e-4, num_steps: int = 5, **kwargs):
        super().__init__(model, **kwargs)
        self.harmful_text = harmful_text
        self.safe_text = safe_text
        self.tokenizer = tokenizer
        self.top_fraction = top_fraction
        self.lr = lr
        self.num_steps = num_steps

    def apply(self, *, harmful_text: str = None, safe_text: str = None,
              tokenizer=None, **kwargs):
        return R.apply_oneshot_safety_patch(
            self.model,
            harmful_text=harmful_text or self.harmful_text,
            safe_text=safe_text or self.safe_text,
            tokenizer=tokenizer or self.tokenizer,
            top_fraction=self.top_fraction,
            lr=self.lr,
            num_steps=self.num_steps,
        )

