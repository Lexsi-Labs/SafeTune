"""Recover runner — layer trainer set."""
from ._base import _RecoverBase
import safetune.recover as R

# ── ReStaTrainer ──────────────────────────────────────────────────────────────
class ReStaTrainer(_RecoverBase):
    """ReSta (DARE task arithmetic): DARE-masked safety vector restoration.

    Args:
        base_model (nn.Module): base model.
        aligned_model (nn.Module): aligned reference.
        alpha: task vector scale. Default 1.0.
        dare: apply DARE masking. Default True.
        dare_seed: DARE random seed. Default 0.
    """

    METHOD = "ReStaTrainer"

    def __init__(self, model=None, *, base_model=None, aligned_model=None,
                 alpha: float = 1.0, dare: bool = True, dare_seed: int = 0,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.base_model = base_model
        self.aligned_model = aligned_model
        self.alpha = alpha
        self.dare = dare
        self.dare_seed = dare_seed

    def apply(self, *, alpha: float = None, dare: bool = None,
              dare_seed: int = None, **kwargs):
        return R.apply_resta(
            self.model,
            base=self.base_model,
            aligned=self.aligned_model,
            alpha=alpha if alpha is not None else self.alpha,
            dare=dare if dare is not None else self.dare,
            dare_seed=dare_seed if dare_seed is not None else self.dare_seed,
        )

# ── SafeLoRATrainer ───────────────────────────────────────────────────────────

class SafeLoRATrainer(_RecoverBase):
    """SafeLoRA: safety subspace LoRA decomposition and merge.

    Args:
        aligned_state_dict (dict): state dict of the aligned model.
        base_state_dict (dict): state dict of the base model.
        alpha: merge coefficient. Default 0.5.
        threshold: safety subspace threshold. Default 0.5.
    """

    METHOD = "SafeLoRATrainer"

    def __init__(self, model=None, *, aligned_state_dict=None, base_state_dict=None,
                 alpha: float = 0.5, threshold: float = 0.5, **kwargs):
        super().__init__(model, **kwargs)
        self.aligned_state_dict = aligned_state_dict
        self.base_state_dict = base_state_dict
        self.alpha = alpha
        self.threshold = threshold

    def apply(self, *, alpha: float = None, threshold: float = None, **kwargs):
        return R.apply_safe_lora(
            self.model,
            aligned_state_dict=self.aligned_state_dict,
            base_state_dict=self.base_state_dict,
            alpha=alpha if alpha is not None else self.alpha,
            threshold=threshold if threshold is not None else self.threshold,
        )

# ── SafeMergeTrainer ──────────────────────────────────────────────────────────

class SafeMergeTrainer(_RecoverBase):
    """SafeMERGE: Fisher-weighted safety merge.

    Args:
        base_model: base model.
        aligned_model: aligned reference.
        threshold: Fisher mask threshold. Default 0.35.
        alpha: merge coefficient. Default 0.5.
    """

    METHOD = "SafeMergeTrainer"

    def __init__(self, model=None, *, base_model=None, aligned_model=None,
                 threshold: float = 0.35, alpha: float = 0.5, **kwargs):
        super().__init__(model, **kwargs)
        self.base_model = base_model
        self.aligned_model = aligned_model
        self.threshold = threshold
        self.alpha = alpha

    def apply(self, *, threshold: float = None, alpha: float = None, **kwargs):
        return R.apply_safemerge(
            self.model,
            base=self.base_model,
            aligned=self.aligned_model,
            threshold=threshold if threshold is not None else self.threshold,
            alpha=alpha if alpha is not None else self.alpha,
        )

# ── SafeDeltaTrainer ──────────────────────────────────────────────────────────

class SafeDeltaTrainer(_RecoverBase):
    """SafeDelta: OBS-style output-weight safety edit.

    Args:
        aligned_model: aligned model (source of safety delta).
        strength: edit strength. Default 0.1.
    """

    METHOD = "SafeDeltaTrainer"

    def __init__(self, model=None, *, aligned_model=None,
                 strength: float = 0.1, **kwargs):
        super().__init__(model, **kwargs)
        self.aligned_model = aligned_model
        self.strength = strength

    def apply(self, *, strength: float = None, **kwargs):
        return R.apply_safe_delta(
            self.model,
            aligned=self.aligned_model,
            strength=strength if strength is not None else self.strength,
        )

# ── QReSafeTrainer ────────────────────────────────────────────────────────────

class QReSafeTrainer(_RecoverBase):
    """QReSafe: selective quantization-aware safety recovery.

    Args:
        mode: quantization mode (``"selective"`` or ``"full"``). Default ``"selective"``.
        quant_bits: quantization bit-width. Default 4.
        calib_inputs: tokenized calibration inputs (list of tensors).
        tau: safety saliency threshold. Default 0.6.
    """

    METHOD = "QReSafeTrainer"

    def __init__(self, model=None, *, mode: str = "selective", quant_bits: int = 4,
                 calib_inputs=None, tau: float = 0.6, **kwargs):
        super().__init__(model, **kwargs)
        self.mode = mode
        self.quant_bits = quant_bits
        self.calib_inputs = calib_inputs
        self.tau = tau

    def apply(self, *, calib_inputs=None, **kwargs):
        return R.apply_qresafe(
            self.model,
            mode=self.mode,
            quant_bits=self.quant_bits,
            calib_inputs=calib_inputs if calib_inputs is not None else self.calib_inputs,
            tau=self.tau,
        )

# ── AAQTrainer ────────────────────────────────────────────────────────────────

class AAQTrainer(_RecoverBase):
    """AAQ: Adversarial Alignment Quantization with APC calibration.

    Args:
        aligned_model_path: HF path/ID for the aligned model.
        base_model_path: HF path/ID for the base model.
        calibration_steps: APC calibration steps. Default 10.
        lr: APC learning rate. Default 5e-6.
        probe_texts: probe calibration texts.
        simulate_quantization: simulate quantization noise. Default True.
        apc_weight: APC regularization weight. Default 0.1.
    """

    METHOD = "AAQTrainer"

    def __init__(self, model=None, *, aligned_model_path: str = None,
                 base_model_path: str = None, calibration_steps: int = 10,
                 lr: float = 5e-6, probe_texts=None,
                 simulate_quantization: bool = True, apc_weight: float = 0.1,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.aligned_model_path = aligned_model_path
        self.base_model_path = base_model_path
        self.calibration_steps = calibration_steps
        self.lr = lr
        self.probe_texts = probe_texts
        self.simulate_quantization = simulate_quantization
        self.apc_weight = apc_weight

    def apply(self, *, probe_texts=None, **kwargs):
        return R.apply_aaq(
            self.model,
            aligned_model_path=self.aligned_model_path or "",
            base_model_path=self.base_model_path or "",
            calibration_steps=self.calibration_steps,
            lr=self.lr,
            probe_texts=probe_texts if probe_texts is not None else self.probe_texts,
            apc_weight=self.apc_weight,
            simulate_quantization=self.simulate_quantization,
        )

# ── RepNoiseRecoverTrainer ────────────────────────────────────────────────────

class RepNoiseRecoverTrainer(_RecoverBase):
    """RepNoise Recover: add representation noise to harm subspace.

    Args:
        harmful_inputs: tokenized harmful calibration inputs.
        noise_scale: noise magnitude. Default 0.01.
        subspace_rank: harm subspace rank. Default 8.
        max_samples: max calibration samples. Default 32.
        seed: random seed. Default 42.
    """

    METHOD = "RepNoiseRecoverTrainer"

    def __init__(self, model=None, *, harmful_inputs=None, noise_scale: float = 0.01,
                 subspace_rank: int = 8, max_samples: int = 32, seed: int = 42,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.harmful_inputs = harmful_inputs
        self.noise_scale = noise_scale
        self.subspace_rank = subspace_rank
        self.max_samples = max_samples
        self.seed = seed

    def apply(self, *, harmful_inputs=None, **kwargs):
        return R.apply_repnoise_recover(
            self.model,
            harmful_inputs=harmful_inputs if harmful_inputs is not None
            else self.harmful_inputs,
            noise_scale=self.noise_scale,
            subspace_rank=self.subspace_rank,
            max_samples=self.max_samples,
            seed=self.seed,
        )

