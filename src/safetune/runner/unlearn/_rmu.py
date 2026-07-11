from ._base import _UnlearnBase
import safetune.unlearn as U


class RMUTrainer(_UnlearnBase):
    METHOD = "RMUTrainer"

    def __init__(self, model=None, *,
                 layer_id: int = 7,
                 update_layer_ids=None,
                 max_num_batches: int = 80,
                 lr: float = 5e-5,
                 alpha: float = 1200.0,
                 steering_coeff: float = 20.0,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.layer_id = layer_id
        self.update_layer_ids = update_layer_ids or [5, 6, 7]
        self.max_num_batches = max_num_batches
        self.lr = lr
        self.alpha = alpha
        self.steering_coeff = steering_coeff

    def unlearn(self, forget, retain, *, frozen_model=None, **kwargs):
        import copy
        if frozen_model is None:
            frozen_model = copy.deepcopy(self.model)
        cfg = U.RMUConfig(
            layer_id=self.layer_id,
            update_layer_ids=self.update_layer_ids,
            max_num_batches=self.max_num_batches,
            lr=self.lr,
            alpha=self.alpha,
            steering_coeff=self.steering_coeff,
        )
        return U.rmu_unlearn(self.model,
                             retain_batches=self._to_device(retain),
                             forget_batches=self._to_device(forget),
                             frozen_model=frozen_model, config=cfg)
