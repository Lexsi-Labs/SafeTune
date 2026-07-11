import torch

from ._base import _UnlearnBase
import safetune.unlearn as U


class SimDPOTrainer(_UnlearnBase):
    METHOD = "SimDPOTrainer"
    USE_LORA = True

    def __init__(self, model=None, *,
                 variant: str = "simdpo_retain",
                 beta: float = 0.1,
                 epochs: int = 5,
                 lr: float = 1e-5,
                 retain_coeff: float = 1.0,
                 safe_refusal: str = "I'm sorry, but I'm unable to assist with that request.",
                 **kwargs):
        super().__init__(model, **kwargs)
        self.variant = variant
        self.beta = beta
        self.epochs = epochs
        self.lr = lr
        self.retain_coeff = retain_coeff
        self.safe_refusal = safe_refusal

    def unlearn(self, forget=None, retain=None, *, forget_pairs=None,
                tokenizer=None, **kwargs):
        # `forget_pairs=` is the pre-fix keyword name — kept as an alias.
        if forget is None:
            forget = forget_pairs
        model = self._wrap_lora(self.model)
        # Accept raw harmful forget batches (consistent with FLAT): SimDPO needs
        # {chosen, rejected} preference pairs, so build them from the raw batches
        # when they are not already in pair form. Pre-built pairs pass through.
        forget = list(forget)
        already_pairs = bool(forget) and isinstance(forget[0], dict) \
            and "chosen" in forget[0] and "rejected" in forget[0]
        if already_pairs:
            forget_pairs = self._to_device(forget)
        else:
            if tokenizer is None:
                from safetune.runner.utils.model_utils import load_tok
                tokenizer = load_tok(self.model_id)
            forget_pairs = self.make_simdpo_pairs(forget, tokenizer)
        cfg = U.SimDPOUnlearnConfig(
            variant=self.variant,
            beta=self.beta,
            epochs=self.epochs,
            lr=self.lr,
            retain_coeff=self.retain_coeff,
        )
        unlearned = U.simdpo_unlearn(model,
                                     forget_batches=forget_pairs,
                                     retain_batches=self._to_device(retain),
                                     config=cfg)
        return self._maybe_merge(unlearned)

    def make_simdpo_pairs(self, harmful_batches, tokenizer):
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = "cpu"

        def _mkbatch(item):
            out = {}
            for k in ("input_ids", "attention_mask", "labels"):
                if k not in item:
                    continue
                v = item[k]
                if isinstance(v, torch.Tensor):
                    out[k] = v.unsqueeze(0).to(device) if v.dim() == 1 else v.to(device)
                elif isinstance(v, list):
                    out[k] = torch.tensor([v]).to(device)
                else:
                    out[k] = v
            return out

        tensorized = [_mkbatch(b) for b in harmful_batches]
        return U.make_simdpo_pairs(tensorized, self.safe_refusal, tokenizer)
