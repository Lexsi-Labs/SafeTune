import torch

from ._base import _UnlearnBase
import safetune.unlearn as U


class FLATTrainer(_UnlearnBase):
    METHOD = "FLATTrainer"
    USE_LORA = True

    def __init__(self, model=None, *,
                 variant: str = "flat_retain",
                 divergence: str = "kl",
                 epochs: int = 5,
                 lr: float = 1e-5,
                 forget_clip: float = 0.5,
                 retain_coeff: float = 1.0,
                 safe_refusal: str = "I'm sorry, but I'm unable to assist with that request.",
                 **kwargs):
        super().__init__(model, **kwargs)
        self.variant = variant
        self.divergence = divergence
        self.epochs = epochs
        self.lr = lr
        self.forget_clip = forget_clip
        self.retain_coeff = retain_coeff
        self.safe_refusal = safe_refusal

    def make_flat_pairs(self, harmful_batches, tokenizer):
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
        pairs = U.make_simdpo_pairs(tensorized, self.safe_refusal, tokenizer)
        good = list(self._to_device([p["chosen"] for p in pairs]))
        forget = list(self._to_device([p["rejected"] for p in pairs]))
        return good, forget

    def unlearn(self, forget, retain, *, good=None, tokenizer=None, **kwargs):
        model = self._wrap_lora(self.model)

        if good is None:
            if tokenizer is None:
                from safetune.runner.utils.model_utils import load_tok
                tokenizer = load_tok(self.model_id)
            good_batches, forget_batches = self.make_flat_pairs(forget, tokenizer)
        else:
            good_batches = self._to_device(good)
            forget_batches = self._to_device(forget)

        cfg = U.FLATConfig(
            variant=self.variant,
            divergence=self.divergence,
            epochs=self.epochs,
            lr=self.lr,
            forget_clip=self.forget_clip,
            retain_coeff=self.retain_coeff,
        )
        unlearned = U.flat_unlearn(model,
                                   forget_batches=forget_batches,
                                   good_batches=good_batches,
                                   retain_batches=self._to_device(retain),
                                   config=cfg)
        return self._maybe_merge(unlearned)
