from ._base import _UnlearnBase
import safetune.unlearn as U


class NPOTrainer(_UnlearnBase):
    METHOD = "NPOTrainer"
    USE_LORA = True

    def __init__(self, model=None, *,
                 variant: str = "npo_grad_diff",
                 beta: float = 0.1,
                 num_epochs: int = 5,
                 lr: float = 1e-5,
                 npo_coeff: float = 1.0,
                 grad_diff_coeff: float = 1.0,
                 kl_coeff: float = 1.0,
                 weight_decay: float = 0.01,
                 max_steps=None,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.variant = variant
        self.beta = beta
        self.num_epochs = num_epochs
        self.lr = lr
        self.npo_coeff = npo_coeff
        self.grad_diff_coeff = grad_diff_coeff
        self.kl_coeff = kl_coeff
        self.weight_decay = weight_decay
        self.max_steps = max_steps

    def unlearn(self, forget, retain, *, reference_model=None, **kwargs):
        model = self._wrap_lora(self.model)
        cfg = U.NPOConfig(
            variant=self.variant,
            beta=self.beta,
            npo_coeff=self.npo_coeff,
            grad_diff_coeff=self.grad_diff_coeff,
            kl_coeff=self.kl_coeff,
            num_epochs=self.num_epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_steps=self.max_steps,
        )
        unlearned = U.npo_unlearn(model,
                                  forget_batches=self._to_device(forget),
                                  retain_batches=self._to_device(retain),
                                  reference=reference_model, config=cfg)
        return self._maybe_merge(unlearned)
