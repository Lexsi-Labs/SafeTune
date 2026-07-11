"""Backward-compat submodule alias — FLAT now lives at ``safetune.unlearn``."""
from safetune.unlearn import flat_fdiv_loss, flat_unlearn, FLATConfig

__all__ = ["flat_fdiv_loss", "flat_unlearn", "FLATConfig"]
