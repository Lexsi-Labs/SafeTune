"""Backward-compat submodule alias — SimDPO now lives at ``safetune.unlearn``."""
from safetune.unlearn import make_simdpo_pairs, simdpo_forget_loss, simdpo_unlearn, SimDPOUnlearnConfig

__all__ = ["make_simdpo_pairs", "simdpo_forget_loss", "simdpo_unlearn", "SimDPOUnlearnConfig"]
