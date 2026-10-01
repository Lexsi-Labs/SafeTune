"""Harden: training-time safety defenses.

``safetune.harden.LisaTrainer`` and the other ``*Trainer`` names are the
high-level trainers, the same classes as ``safetune.runner.harden``::

    trainer = harden.LisaTrainer(model, tokenizer, lisa_rho=0.1)
    checkpoint = trainer.train(train_dataset)

The ``transformers.Trainer`` subclasses they run are exported as ``*HFTrainer``
(``LisaHFTrainer`` with ``LisaConfig``, ...; DOOR's is ``SafetyDOORTrainer``).
Use those to drive your own HF training loop, with your own ``TrainingArguments``
and data loaders, no LoRA wrapping and no merged checkpoint.
"""

from .safegrad import SafeGradHFTrainer, SafeGradConfig
from .lisa import LisaHFTrainer, LisaConfig
from .asft import AsFTHFTrainer, AsFTConfig
from .star_dss import STARDSSHFTrainer, STARDSSConfig
from .sap import SAPHFTrainer, SAPConfig
from .sppft import SPPFTHFTrainer, SPPFTConfig
from .ema import EMACallback
from .door import SafetyDOORTrainer, DOORConfig
from .derta import DeRTaHFTrainer, DeRTaConfig
from .cst import CSTConfig
from .vaccine import VaccineConfig, vaccine_loss
from .tvaccine import TVaccineConfig, tvaccine_loss
from .booster import (BoosterConfig, booster_project, collect_harmful_gradient,
                      booster_simulated_perturbation, booster_perturb_weights,
                      booster_restore_weights)
from .repnoise import RepNoiseHFTrainer, RepNoiseConfig
from .seam import SEAMHFTrainer, SEAMConfig
from .ctrap import CTRAPHFTrainer, CTRAPConfig
from .seal import SEALHFTrainer, SEALConfig
from .constrained_sft import ConstrainedSFTHFTrainer, ConstrainedSFTConfig
from .lox_harden import LoXHardenConfig, apply_lox_harden
from .salora import SaLoRAConfig, compute_safety_subspace, project_lora_step
from .tar import TARConfig, tar_outer_loss
from .asrt import ASRTCallback, ASRTConfig
from .mart import MARTConfig

try:
    from .deeprefusal import DeepRefusalConfig
except Exception:  # pragma: no cover
    DeepRefusalConfig = None  # type: ignore[assignment]

try:
    from .lookahead import LookAheadHFTrainer, LookAheadConfig
except Exception:  # pragma: no cover
    LookAheadHFTrainer = None  # type: ignore[assignment]
    LookAheadConfig = None  # type: ignore[assignment]

try:
    from .surgery import SurgeryHFTrainer, SurgeryConfig
except Exception:  # pragma: no cover
    SurgeryHFTrainer = None  # type: ignore[assignment]
    SurgeryConfig = None  # type: ignore[assignment]

try:
    from .antibody import AntibodyConfig
except Exception:  # pragma: no cover
    AntibodyConfig = None  # type: ignore[assignment]

# The high-level trainers live in safetune.runner.harden, which imports this
# package, so they are resolved on first access.
_RUNNER_NAMES = (
    "PlainSFTTrainer", "SafeGradTrainer", "LisaTrainer", "SPPFTTrainer",
    "LookAheadTrainer", "STARDSSTrainer", "DeRTaTrainer", "AsFTTrainer",
    "SAPTrainer", "SurgeryTrainer", "BoosterTrainer", "VaccineTrainer",
    "TVaccineTrainer", "RepNoiseTrainer", "CTRAPTrainer", "SEAMTrainer",
    "DOORTrainer", "TARTrainer", "SaLoRATrainer", "SEALTrainer",
    "ConstrainedSFTTrainer", "LoXHardenTrainer", "load_harden_data",
    # previously "Python API only" — the runner adapters in _dpo_adversarial
    "CSTTrainer", "MARTTrainer", "DeepRefusalTrainer", "AntibodyTrainer",
)


def __getattr__(name):
    if name in _RUNNER_NAMES:
        from safetune.runner import harden as _runner
        return getattr(_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_RUNNER_NAMES))


__all__ = [
    *_RUNNER_NAMES,
    # transformers.Trainer subclasses
    "SafeGradHFTrainer", "SafeGradConfig",
    "LisaHFTrainer", "LisaConfig",
    "AsFTHFTrainer", "AsFTConfig",
    "STARDSSHFTrainer", "STARDSSConfig",
    "SAPHFTrainer", "SAPConfig",
    "SPPFTHFTrainer", "SPPFTConfig",
    "EMACallback",
    "SafetyDOORTrainer", "DOORConfig",
    "DeRTaHFTrainer", "DeRTaConfig",
    "CSTConfig",
    "LookAheadHFTrainer", "LookAheadConfig",
    "SurgeryHFTrainer", "SurgeryConfig",
    "AntibodyConfig",
    "MARTConfig",
    "ASRTCallback", "ASRTConfig",
    "DeepRefusalConfig",
    # Vaccine family
    "VaccineConfig", "vaccine_loss",
    "TVaccineConfig", "tvaccine_loss",
    # Booster family
    "BoosterConfig", "booster_project", "collect_harmful_gradient",
    "booster_simulated_perturbation", "booster_perturb_weights",
    "booster_restore_weights",
    # Representation-space defenses
    "RepNoiseHFTrainer", "RepNoiseConfig",
    "CTRAPHFTrainer", "CTRAPConfig",
    "SEAMHFTrainer", "SEAMConfig",
    # Data-selection defense
    "SEALHFTrainer", "SEALConfig",
    # First-token constraint
    "ConstrainedSFTHFTrainer", "ConstrainedSFTConfig",
    # Pre-FT subspace extrapolation
    "LoXHardenConfig", "apply_lox_harden",
    # SaLoRA
    "SaLoRAConfig", "compute_safety_subspace", "project_lora_step",
    # TAR
    "TARConfig", "tar_outer_loss",
]