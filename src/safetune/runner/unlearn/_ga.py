from ._base import _UnlearnBase
import safetune.unlearn as U


class GradientAscentTrainer(_UnlearnBase):
    METHOD = "GradientAscentTrainer"

    def __init__(self, model=None, *,
                 forget_loss: str = "grad_ascent",
                 epochs: int = 5,
                 max_steps: int = 200,
                 lr: float = 1e-5,
                 forget_clip: float = 0.5,
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
            forget_clip=self.forget_clip,
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
                 forget_clip: float = 0.5,
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
            forget_clip=self.forget_clip,
        )
        return U.gradient_ascent_unlearn(self.model,
                                         forget_batches=self._to_device(forget),
                                         retain_batches=self._to_device(retain),
                                         config=cfg)
