"""Safety metrics re-export."""

from __future__ import annotations

try:
    from safetune.core.eval.metrics.safety import *  # noqa: F401,F403
    from safetune.core.eval.metrics.safety import (  # noqa: F401
        HarmfulnessMetric,
        OverRefusalMetric,
        CapabilityRegressionMetric,
        JailbreakSuccessMetric,
        compute_safety_suite,
        compute_safety_gates,
    )
    _IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover
    _IMPORT_ERROR = _e
    HarmfulnessMetric = None  # type: ignore[assignment]
    OverRefusalMetric = None  # type: ignore[assignment]
    CapabilityRegressionMetric = None  # type: ignore[assignment]
    JailbreakSuccessMetric = None  # type: ignore[assignment]
    compute_safety_suite = None  # type: ignore[assignment]
    compute_safety_gates = None  # type: ignore[assignment]
