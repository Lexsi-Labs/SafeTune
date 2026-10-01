from ._base import _UnlearnBase
import safetune.unlearn as U
from safetune.config import get_config


def _forget_clip(value):
    """``forget_clip`` caps the forget CE, so ascent stops once the loss passes
    it. ``None``: no cap, TOFU's pure ascent (the paper setting). The old default
    0.5 sat below where real forget sets start (BeaverTails completions: ~2.4),
    so the clamp zeroed the forget gradient from step one and GA did nothing;
    ``safetune.configure(legacy_ga_forget_clip=True)`` restores it. Under that
    switch, ``forget_clip=float("inf")`` turns the cap off."""
    if value is None and get_config().legacy_ga_forget_clip:
        return 0.5
    return value


class GradientAscentTrainer(_UnlearnBase):
    METHOD = "GradientAscentTrainer"

    def __init__(self, model=None, *,
                 forget_loss: str = "grad_ascent",
                 epochs: int = 5,
                 max_steps: int = 200,
                 lr: float = 1e-5,
                 forget_clip: float = None,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.forget_loss = forget_loss
        self.epochs = epochs
        self.max_steps = max_steps
        self.lr = lr
        self.forget_clip = forget_clip

    def unlearn(self, forget, retain, **kwargs):
        cfg = U.GradientAscentConfig(
            forget_loss=self.forget_loss,
            epochs=self.epochs,
            max_steps=self.max_steps,
            lr=self.lr,
            forget_clip=_forget_clip(self.forget_clip),
        )
        return U.gradient_ascent_unlearn(self.model,
                                         forget_batches=self._to_device(forget),
                                         retain_batches=self._to_device(retain),
                                         config=cfg)


class GradDiffTrainer(_UnlearnBase):
    METHOD = "GradDiffTrainer"
    # GradDiff trains full weights (unlearn() enables grad on all params and
    # never wraps LoRA) — unlike its NPO/FLAT/SimDPO siblings.
    USE_LORA = False

    def __init__(self, model=None, *,
                 epochs: int = 5,
                 max_steps: int = 200,
                 lr: float = 1e-5,
                 forget_clip: float = None,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.epochs = epochs
        self.max_steps = max_steps
        self.lr = lr
        self.forget_clip = forget_clip

    def unlearn(self, forget, retain, **kwargs):
        for param in self.model.parameters():
            param.requires_grad = True
            
        cfg = U.GradientAscentConfig(
            forget_loss="grad_diff",
            epochs=self.epochs,
            max_steps=self.max_steps,
            lr=self.lr,
            forget_clip=_forget_clip(self.forget_clip),
        )
        return U.gradient_ascent_unlearn(self.model,
                                         forget_batches=self._to_device(forget),
                                         retain_batches=self._to_device(retain),
                                         config=cfg)
