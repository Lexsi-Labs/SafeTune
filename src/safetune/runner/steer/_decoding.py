from ._base import _SteerBase
import safetune.steer as S


class SafeDecodingTrainer(_SteerBase):
    METHOD = "SafeDecodingTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 alpha: float = 2.0, window: int = 5,
                 banned_tokens=None, expert_model=None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.alpha = alpha
        self.window = window
        self.banned_tokens = banned_tokens
        self.expert_model = expert_model

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if self.expert_model is None:
            raise ValueError(
                "SafeDecoding requires a distinct safety-expert model on the "
                "HF path; pass expert_model=... or use the vLLM backend. "
                "Using the target model as its own expert makes the "
                "base/expert blend an identity (no-op) processor."
            )
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        cfg = S.SafeDecodingConfig(alpha=self.alpha, first_m=self.window,
                                   banned_tokens=self.banned_tokens)
        return S.SafeDecodingProcessor(self.expert_model, self.tok,
                                       config=cfg, target=self.model)

    def make_processor(self, harmful=None, harmless=None, *, calib_n: int = None):
        return self._do_calibrate(harmful=harmful, harmless=harmless, calib_n=calib_n)


class ContrastiveDecodingTrainer(_SteerBase):
    METHOD = "ContrastiveDecodingTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 alpha: float = 1.0, temperature: float = 1.0,
                 weak_model=None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.weak_model = weak_model

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if self.weak_model is None:
            raise ValueError(
                "ContrastiveDecoding requires a distinct weak/amateur model "
                "on the HF path; pass weak_model=... or use the vLLM backend. "
                "Contrasting the target model against itself yields "
                "(1-alpha)*logits, which leaves the greedy argmax unchanged "
                "(no-op processor)."
            )
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        cfg = S.ContrastiveDecodingConfig(alpha=self.alpha)
        return S.ContrastiveDecodingProcessor(self.weak_model, self.tok, config=cfg, target=self.model)

    def make_processor(self):
        return self._do_calibrate()


class ProxyTuningTrainer(_SteerBase):
    METHOD = "ProxyTuningTrainer"

    """Proxy Tuning: ``model`` is the large base model (and the model that generates); the delta
    ``proxy_tuned_model - proxy_base_model`` of the small pair is added to its logits."""

    def __init__(self, model=None, tokenizer=None, *,
                 proxy_tuned_model=None, proxy_base_model=None, alpha: float = 1.0,
                 proxy_model=None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        if proxy_model is not None:
            raise ValueError(
                "ProxyTuningTrainer: proxy_model is gone. Pass proxy_tuned_model (the small tuned model) and "
                "proxy_base_model (the small untuned model); `model` is the large base. The old proxy_model "
                "was the untuned side of the delta and `model` the tuned one, so mapping it onto either new "
                "argument would flip or break the correction."
            )
        self.proxy_tuned_model = proxy_tuned_model
        self.proxy_base_model = proxy_base_model
        self.alpha = alpha

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if self.proxy_tuned_model is None or self.proxy_base_model is None:
            raise ValueError(
                "ProxyTuningTrainer needs both proxy_tuned_model (the small tuned model) and "
                "proxy_base_model (the small untuned model); `model` is the large base. Proxy Tuning adds "
                "tuned - base to the large model's logits and cannot run with fewer than three models."
            )
        if self.proxy_tuned_model is self.proxy_base_model:
            raise ValueError(
                "ProxyTuningTrainer: proxy_tuned_model and proxy_base_model are the same object, so "
                "tuned - base is zero and the processor would change nothing. Pass the small tuned model "
                "and the small untuned model."
            )
        cfg = S.ProxyTuningConfig(scale=self.alpha)
        return S.ProxyTuningProcessor(self.proxy_tuned_model, self.proxy_base_model,
                                       self.tok, config=cfg, target=self.model)

    def make_processor(self):
        return self._do_calibrate()


class NudgingTrainer(_SteerBase):
    METHOD = "NudgingTrainer"

    """Nudging: ``model`` is the base model that generates; ``guide_model`` is the small aligned model
    that takes over the steps where the base model's top-1 probability is below the threshold."""

    def __init__(self, model=None, tokenizer=None, *,
                 guide_model=None,
                 nudge_strength: float = 1.0,
                 top_prob_thres: float = None,
                 safe_tokens=None, unsafe_tokens=None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.guide_model = guide_model
        self.nudge_strength = nudge_strength
        self.top_prob_thres = top_prob_thres
        self.safe_tokens = safe_tokens
        self.unsafe_tokens = unsafe_tokens

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if self.guide_model is None:
            raise ValueError(
                "NudgingTrainer needs guide_model, the aligned model that nudges the base model. With the "
                "base model as its own guide the hand-over would be an identity (no-op) processor."
            )
        if self.guide_model is self.model:
            raise ValueError(
                "NudgingTrainer: guide_model is the target model itself, so every hand-over would give the "
                "model its own logits (an identity processor). Pass a distinct aligned guide model."
            )
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        cfg_kwargs = ({} if self.top_prob_thres is None
                      else {"top_prob_thres": self.top_prob_thres})
        cfg = S.NudgingConfig(
            nudge_strength=self.nudge_strength,
            safe_tokens=self.safe_tokens,
            unsafe_tokens=self.unsafe_tokens,
            **cfg_kwargs,
        )
        return S.NudgingProcessor(self.guide_model, self.tok, config=cfg, target=self.model)

    def make_processor(self, harmful=None, harmless=None, *, calib_n: int = None):
        return self._do_calibrate(harmful=harmful, harmless=harmless, calib_n=calib_n)
