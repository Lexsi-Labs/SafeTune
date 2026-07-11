"""Advanced utilities: steering vectors, subspaces, legacy orchestrators."""
from __future__ import annotations

try:
    from .vectors import SteeringVectorExtractor
    _VECTORS_OK = True
except Exception as _e:
    _VECTORS_OK = False

try:
    from .subspace import compute_safety_subspace, compute_refusal_direction
    _SUBSPACE_OK = True
except Exception as _e:
    _SUBSPACE_OK = False

try:
    from .config import UnifiedSafetyConfig
    _CONFIG_OK = True
except Exception as _e:
    _CONFIG_OK = False

__all__ = []

if _VECTORS_OK:
    __all__ += ["SteeringVectorExtractor"]
if _SUBSPACE_OK:
    __all__ += ["compute_safety_subspace", "compute_refusal_direction"]
if _CONFIG_OK:
    __all__ += ["UnifiedSafetyConfig"]
