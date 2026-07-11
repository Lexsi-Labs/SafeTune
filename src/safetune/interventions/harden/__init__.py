"""Training-time safety defenses (Trainer subclasses)."""

from .safegrad import SafeGradTrainer, SafeGradConfig
from .lisa import LisaTrainer, LisaConfig
from .asft import AsFTTrainer, AsFTConfig
from .star_dss import STARDSSTrainer, STARDSSConfig
from .sap import SAPTrainer, SAPConfig
from .sppft import SPPFTTrainer, SPPFTConfig
from .ema import EMACallback
from .door import SafetyDOORTrainer as DOORTrainer, DOORConfig
from .derta import DeRTaTrainer, DeRTaConfig
from .cst import CSTTrainer, CSTConfig
from .vaccine import VaccineConfig, vaccine_loss
from .tvaccine import TVaccineConfig, tvaccine_loss
from .booster import (BoosterConfig, booster_project, collect_harmful_gradient,
                      booster_simulated_perturbation, booster_perturb_weights,
                      booster_restore_weights)
from .repnoise import RepNoiseTrainer, RepNoiseConfig
from .seam import SEAMTrainer, SEAMConfig
from .ctrap import CTRAPTrainer, CTRAPConfig
from .seal import SEALTrainer, SEALConfig
from .constrained_sft import ConstrainedSFTTrainer, ConstrainedSFTConfig
from .lox_harden import LoXHardenConfig, apply_lox_harden
from .salora import SaLoRAConfig, compute_safety_subspace, project_lora_step
from .tar import TARConfig, tar_outer_loss
from .asrt import ASRTCallback, ASRTConfig
from .mart import MARTTrainer, MARTConfig

try:
    from .deeprefusal import DeepRefusalTrainer, DeepRefusalConfig
except Exception:  # pragma: no cover
    DeepRefusalTrainer = None  # type: ignore[assignment]
    DeepRefusalConfig = None  # type: ignore[assignment]

try:
    from .lookahead import LookAheadTrainer, LookAheadConfig
except Exception:  # pragma: no cover
    LookAheadTrainer = None  # type: ignore[assignment]
    LookAheadConfig = None  # type: ignore[assignment]

try:
    from .surgery import SurgeryTrainer, SurgeryConfig
except Exception:  # pragma: no cover
    SurgeryTrainer = None  # type: ignore[assignment]
    SurgeryConfig = None  # type: ignore[assignment]

try:
    from .antibody import AntibodyTrainer, AntibodyConfig
except Exception:  # pragma: no cover
    AntibodyTrainer = None  # type: ignore[assignment]
    AntibodyConfig = None  # type: ignore[assignment]

__all__ = [
    # Core trainers
    "SafeGradTrainer", "SafeGradConfig",
    "LisaTrainer", "LisaConfig",
    "AsFTTrainer", "AsFTConfig",
    "STARDSSTrainer", "STARDSSConfig",
    "SAPTrainer", "SAPConfig",
    "SPPFTTrainer", "SPPFTConfig",
    "EMACallback",
    "DOORTrainer", "DOORConfig",
    "DeRTaTrainer", "DeRTaConfig",
    "CSTTrainer", "CSTConfig",
    "LookAheadTrainer", "LookAheadConfig",
    "SurgeryTrainer", "SurgeryConfig",
    "AntibodyTrainer", "AntibodyConfig",
    "MARTTrainer", "MARTConfig",
    "ASRTCallback", "ASRTConfig",
    "DeepRefusalTrainer", "DeepRefusalConfig",
    # Vaccine family
    "VaccineConfig", "vaccine_loss",
    "TVaccineConfig", "tvaccine_loss",
    # Booster family
    "BoosterConfig", "booster_project", "collect_harmful_gradient",
    "booster_simulated_perturbation", "booster_perturb_weights",
    "booster_restore_weights",
    # Representation-space defenses
    "RepNoiseTrainer", "RepNoiseConfig",
    "CTRAPTrainer", "CTRAPConfig",
    "SEAMTrainer", "SEAMConfig",
    # Data-selection defense
    "SEALTrainer", "SEALConfig",
    # First-token constraint
    "ConstrainedSFTTrainer", "ConstrainedSFTConfig",
    # Pre-FT subspace extrapolation
    "LoXHardenConfig", "apply_lox_harden",
    # SaLoRA
    "SaLoRAConfig", "compute_safety_subspace", "project_lora_step",
    # TAR
    "TARConfig", "tar_outer_loss",
]