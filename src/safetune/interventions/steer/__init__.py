"""Inference-time safety steering / representation control wrappers."""

from .adasteer import AdaSteerModel
from .safeswitch import SafeSwitchModel
from .scans import SCANSModel
from .sta import STAModel
from .circuit_breaker import CircuitBreakerModel
from .repbend import RepBendModel
from .tar import TARModel
from .rrfa import RRFAEnsemble
from .refusal_direction import (
    RefusalDirectionConfig,
    RefusalDirectionModel,
    extract_refusal_direction,
    orthogonalize_weights,
    restore_weights,
)
from .caa import CAAConfig, CAAModel, extract_caa_vectors
from .cast import CASTModel, CASTCondition, fit_cast_condition, fit_cast_probe
from .probe_guard import (
    LinearProbe,
    LinearProbeConfig,
    LinearProbeGuardModel,
    fit_linear_probe,
)
from .circuit_breakers_rr import CircuitBreakerRRModel, CircuitBreakerRRConfig
from .decoding import (
    ContrastiveDecodingConfig,
    ContrastiveDecodingProcessor,
    ProxyTuningConfig,
    ProxyTuningProcessor,
    SafeDecodingConfig,
    SafeDecodingProcessor,
    NudgingConfig,
    NudgingProcessor,
)

# ── Inference backends + unified runner ─────────────────────────────────────
# `run` dispatches generation across the hf / vllm-hook / vllm-logits backends;
# see `safetune.steer.backends`.
from . import backends
from .backends import run

try:
    from .safesteer import SafeSteerModel
except Exception:  # pragma: no cover
    SafeSteerModel = None  # type: ignore[assignment]

try:
    from .alphasteer import AlphaSteerModel
except Exception:  # pragma: no cover
    AlphaSteerModel = None  # type: ignore[assignment]

__all__ = [
    "run",
    "backends",
    "AdaSteerModel",
    "SafeSwitchModel",
    "SCANSModel",
    "STAModel",
    "CircuitBreakerModel",
    "RepBendModel",
    "TARModel",
    "RRFAEnsemble",
    "SafeSteerModel",
    "AlphaSteerModel",
    "RefusalDirectionConfig",
    "RefusalDirectionModel",
    "extract_refusal_direction",
    "orthogonalize_weights",
    "restore_weights",
    "CAAConfig",
    "CAAModel",
    "extract_caa_vectors",
    "CASTModel",
    "CASTCondition",
    "fit_cast_condition",
    "fit_cast_probe",
    "LinearProbe",
    "LinearProbeConfig",
    "LinearProbeGuardModel",
    "fit_linear_probe",
    "CircuitBreakerRRModel",
    "CircuitBreakerRRConfig",
    "ContrastiveDecodingConfig", "ContrastiveDecodingProcessor",
    "ProxyTuningConfig", "ProxyTuningProcessor",
    "SafeDecodingConfig", "SafeDecodingProcessor",
    "NudgingConfig", "NudgingProcessor",
]
