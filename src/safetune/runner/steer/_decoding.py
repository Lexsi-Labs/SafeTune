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
                                       prompt_length=0, config=cfg)

    def make_processor(self, harmful=None, harmless=None, *, calib_n: int = 256):
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
        return S.ContrastiveDecodingProcessor(self.weak_model, self.tok, config=cfg)

    def make_processor(self):
        return self._do_calibrate()


class ProxyTuningTrainer(_SteerBase):
    METHOD = "ProxyTuningTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 proxy_model=None, alpha: float = 1.0, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.proxy_model = proxy_model
        self.alpha = alpha

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        cfg = S.ProxyTuningConfig(scale=self.alpha)
        return S.ProxyTuningProcessor(self.model, self.proxy_model,
                                       self.tok, config=cfg)

    def make_processor(self):
        return self._do_calibrate()


class NudgingTrainer(_SteerBase):
    METHOD = "NudgingTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 nudge_strength: float = 1.0,
                 top_prob_thres: float = None,
                 safe_tokens=None, unsafe_tokens=None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.nudge_strength = nudge_strength
        self.top_prob_thres = top_prob_thres
        self.safe_tokens = safe_tokens
        self.unsafe_tokens = unsafe_tokens

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
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
        return S.NudgingProcessor(self.model, self.tok, config=cfg)

    def make_processor(self, harmful=None, harmless=None, *, calib_n: int = 256):
        return self._do_calibrate(harmful=harmful, harmless=harmless, calib_n=calib_n)
