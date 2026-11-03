"""LookAhead Tuning config. The collator lives in ``safetune.interventions.harden.lookahead``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

try:
    import torch
    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = _e


@dataclass
class LookAheadConfig:
    prefix_mode: str = "virtual"
    prefix_length: int = 10
    prefix_token_ids: Optional[List[int]] = None


__all__ = ["LookAheadConfig"]
