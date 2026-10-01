import torch

from ._base import _SteerBase
from safetune._refusal_helpers import _encode_prompts, _get_decoder_layers
from safetune.config import get_config
import safetune.steer as S

# The absolute layer defaults below (CAA/CAST 14-18, LinearProbeGuard 15,
# SafeSwitch 16, AlphaSteer 10-19) were set for 32-layer models (Llama-3.1-8B).
_REF_DEPTH = 32


def _default_layers(model, layers) -> list:
    """The default band ``layers`` (contiguous indices on a 32-layer model) at the
    same relative depth in ``model``, scaling its end points like SPPFT's
    ``_middle_layer_indices``. Unchanged at 32 layers or with
    ``configure(legacy_steer_layers=True)``."""
    n = len(_get_decoder_layers(model))
    if not n or get_config().legacy_steer_layers:
        return list(layers)
    lo, hi = (min(n - 1, round(i * n / _REF_DEPTH)) for i in (layers[0], layers[-1]))
    return list(range(lo, hi + 1))


class RefusalDirectionTrainer(_SteerBase):
    METHOD = "RefusalDirectionTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 layers=None, alpha: float = 20.0,
                 orthogonalize: bool = False, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.layers = layers
        self.alpha = alpha
        self.orthogonalize = orthogonalize

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  alpha: float = None, **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        direction, _, _ = S.extract_refusal_direction(self.model, self.tok, harmful, harmless)
        strength = alpha if alpha is not None else self.alpha
        return S.RefusalDirectionModel(
            self.model,
            direction=direction,
            mode="steer",
            strength=strength,
            layers=self.layers,
        )


class CAATrainer(_SteerBase):
    METHOD = "CAATrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 target_layers=None, pool_method: str = "mean",
                 normalize: bool = True, multiplier: float = 20.0, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.target_layers = target_layers  # None -> 14-18 scaled to depth
        self.pool_method = pool_method
        self.normalize = normalize
        self.multiplier = multiplier

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        cfg = S.CAAConfig(
            target_layers=self.target_layers or _default_layers(self.model, range(14, 19)),
            pool_method=self.pool_method,
            normalize=self.normalize,
        )
        vectors = S.extract_caa_vectors(self.model, self.tok, harmful, harmless, cfg)
        return S.CAAModel(self.model, vectors=vectors, strength=self.multiplier)


class LinearProbeGuardTrainer(_SteerBase):
    METHOD = "LinearProbeGuardTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 layer: int = None, threshold: float = 0.5, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.layer = layer  # None -> 15 scaled to depth
        self.threshold = threshold

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        layer = self.layer if self.layer is not None else _default_layers(self.model, [15])[0]
        cfg = S.LinearProbeConfig(pick_layer=layer, threshold=self.threshold)
        probe = S.fit_linear_probe(self.model, self.tok, harmful, harmless, cfg)
        return S.LinearProbeGuardModel(self.model, self.tok, probe)


class SCANSTrainer(_SteerBase):
    METHOD = "SCANSTrainer"

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        wrapped = S.SCANSModel(self.model, tokenizer=self.tok)
        wrapped.fit(harmful, harmless)
        return wrapped


class STATrainer(_SteerBase):
    METHOD = "STATrainer"

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        return S.STAModel(self.model)


class CircuitBreakerRRTrainer(_SteerBase):
    METHOD = "CircuitBreakerRRTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 rr_layers=None, target_layers=None,
                 threshold: float = 0.5, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.rr_layers = rr_layers
        self.target_layers = target_layers
        self.threshold = threshold

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        cfg = S.CircuitBreakerRRConfig(threshold=self.threshold)
        _, _, directions = S.extract_refusal_direction(self.model, self.tok, harmful, harmless)
        cb = S.CircuitBreakerRRModel(self.model, directions=directions, config=cfg)
        cb.install()
        return cb


class CASTTrainer(_SteerBase):
    METHOD = "CASTTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 behavior_layers=None, condition_layers=None,
                 alpha: float = 1.0, chat_template: bool = None,
                 per_prompt_gate: bool = None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.behavior_layers = behavior_layers  # None -> 14-18 scaled to depth
        self.condition_layers = condition_layers
        self.alpha = alpha
        # None: True, or False under configure(legacy_cast_gate=True) (the old
        # gate: fit on raw prompts, first prompt of a batch decides for all).
        self.chat_template = chat_template
        self.per_prompt_gate = per_prompt_gate

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        caa_cfg = S.CAAConfig(target_layers=self.behavior_layers
                              or _default_layers(self.model, range(14, 19)))
        vectors = S.extract_caa_vectors(self.model, self.tok, harmful, harmless, caa_cfg)
        condition = S.fit_cast_condition(
            self.model, harmful, harmless, self.tok,
            candidate_layers=self.condition_layers,
            chat_template=self.chat_template,
        )
        return S.CASTModel(
            self.model,
            steering_vectors=vectors,
            condition=condition,
            alpha=self.alpha,
            per_prompt_gate=self.per_prompt_gate,
        )


class AdaSteerTrainer(_SteerBase):
    METHOD = "AdaSteerTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 alpha: float = 15.0, layers=None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.alpha = alpha
        self.layers = layers

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        import logging
        logger = logging.getLogger(__name__)

        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        _, _, directions = S.extract_refusal_direction(self.model, self.tok, harmful, harmless)

        # Pick RD/HD probe layers that actually exist in this model, keeping
        # AdaSteer's defaults (8 / 13) when available.
        layer_keys = sorted(directions)
        rd_probe = 8 if 8 in directions else layer_keys[len(layer_keys) // 3]
        hd_probe = 13 if 13 in directions else layer_keys[len(layer_keys) // 2]

        # Collect per-prompt last-token activations at the probe layers
        # BEFORE the AdaSteerModel installs its steering hooks, so the
        # calibration features are un-steered.
        from safetune.core.runtime.inference.vector_extraction import (
            SteeringVectorExtractor, VectorExtractionConfig,
        )
        ve_cfg = VectorExtractionConfig(
            target_layers=sorted({rd_probe, hd_probe}),
            pool_method="last_token",
        )
        extractor = SteeringVectorExtractor(self.model, self.tok, ve_cfg)
        harmful_acts = extractor._collect_activations(harmful, desc="adasteer harmful")
        benign_acts = extractor._collect_activations(harmless, desc="adasteer benign")

        # The trainer only has a harmful/harmless contrast set, not the
        # adversarial-vs-benign pseudo pair AdaSteer uses for a distinct
        # Harmfulness Direction; AdaSteerModel falls back to HD = copy of RD
        # (single-law mode) when harmfulness_direction is omitted.
        logger.warning(
            "AdaSteer: no adversarial/benign pseudo-pair set available on the "
            "trainer calibrate path; the Harmfulness Direction falls back to a "
            "copy of the Rejection Direction (single-law mode). The adaptive "
            "logistic laws are still fitted per input."
        )

        wrapped = S.AdaSteerModel(
            self.model,
            rejection_direction=directions,
            target_layers=self.layers,
            base_multiplier=self.alpha,
            rd_probe_layer=rd_probe,
            hd_probe_layer=hd_probe,
        )
        # Fit the logistic R-Law/H-Law so _coeff() gating is per-input
        # adaptive instead of the uniform-max fallback.
        wrapped.fit_adaptive(harmful_acts, benign_acts)
        return wrapped


class SafeSwitchTrainer(_SteerBase):
    METHOD = "SafeSwitchTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 gate_layer: int = None, threshold: float = 0.5,
                 chat_template: bool = True, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.gate_layer = gate_layer  # None -> 16 scaled to depth
        self.threshold = threshold
        self.chat_template = chat_template  # fit the prober on prompts formatted as generation sees them

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        from safetune.core.runtime.inference.safeswitch import SafetyProber
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        layer = self.gate_layer if self.gate_layer is not None else _default_layers(self.model, [16])[0]
        # Fit the stage-1 prober on the calibration prompts. Without this the
        # wrapper's prober was never trained, so it never fired on any model.
        dev = next(self.model.parameters()).device
        enc = lambda prompts: [_encode_prompts(self.tok, [p], self.chat_template,
                                               return_tensors="pt").to(dev) for p in prompts]
        # Vision-language configs (Aya Vision, North) keep hidden_size on the text config.
        hidden_size = self.model.config.get_text_config().hidden_size
        prober = SafetyProber(hidden_size=hidden_size, layer_idx=layer)
        prober.train(self.model, safe_inputs=enc(harmless), unsafe_inputs=enc(harmful))
        return S.SafeSwitchModel(
            self.model,
            prober=prober,
            probe_layer=layer,
            hidden_size=hidden_size,
            unsafe_threshold=self.threshold,
        )


class CircuitBreakerTrainer(_SteerBase):
    METHOD = "CircuitBreakerTrainer"

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        return S.CircuitBreakerModel(self.model)


class RepBendTrainer(_SteerBase):
    METHOD = "RepBendTrainer"

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        return S.RepBendModel(self.model)


class TARSteerTrainer(_SteerBase):
    METHOD = "TARSteerTrainer"

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        return S.TARModel(self.model)


class RRFAEnsembleTrainer(_SteerBase):
    METHOD = "RRFAEnsembleTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 member_models=None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.member_models = member_models or []

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        return S.RRFAEnsemble(self.model)


class SafeSteerTrainer(_SteerBase):
    METHOD = "SafeSteerTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 alpha: float = 15.0, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.alpha = alpha

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        _, layer_id, directions = S.extract_refusal_direction(self.model, self.tok, harmful, harmless)
        category_vectors = {"default": directions[layer_id]}
        return S.SafeSteerModel(self.model, category_vectors, layer_id=layer_id, alpha=self.alpha)


class AlphaSteerTrainer(_SteerBase):
    METHOD = "AlphaSteerTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 alpha: float = 20.0, layers=None,
                 legacy_alphasteer_layers: bool = None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.alpha = alpha
        self.layers = layers
        # True: hook the matrix fitted on layers[i] at decoder layer i (the old
        # mapping: fitted on 10-19, applied on 0-9). None: the configure() key.
        self.legacy_alphasteer_layers = legacy_alphasteer_layers

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        legacy = (self.legacy_alphasteer_layers if self.legacy_alphasteer_layers is not None
                  else get_config().legacy_alphasteer_layers)
        from safetune.core.runtime.inference.vector_extraction import (
            SteeringVectorExtractor, VectorExtractionConfig,
        )
        target = self.layers or _default_layers(self.model, range(10, 20))
        ve_cfg = VectorExtractionConfig(target_layers=target, pool_method="last_token")
        extractor = SteeringVectorExtractor(self.model, self.tok, ve_cfg)
        harmful_acts = extractor._collect_activations(harmful)
        harmless_acts = extractor._collect_activations(harmless)
        layer_keys = sorted(set(harmful_acts) & set(harmless_acts))
        if legacy:
            stack = lambda acts: torch.stack([acts[l].float() for l in layer_keys], dim=1)
            layers = list(range(len(layer_keys)))
        else:
            # AlphaSteerModel reads dim 1 as the decoder index, so put each
            # layer's activations at its own index (unused slots stay zero).
            def stack(acts):
                first = acts[layer_keys[0]]
                out = torch.zeros(first.shape[0], layer_keys[-1] + 1, first.shape[-1])
                for l in layer_keys:
                    out[:, l] = acts[l].float()
                return out
            layers = layer_keys
        return S.AlphaSteerModel(
            self.model,
            harmful_activations=stack(harmful_acts),
            benign_activations=stack(harmless_acts),
            layers=layers,
            strength=self.alpha,
        )
