"""SCRUB-style selective unlearning, exposed under the Recover pillar.

Thin re-export so users can call ``safetune.recover.scrub_unlearn(...)``.
"""
from __future__ import annotations

from safetune.interventions.unlearn import SCRUBConfig, scrub_unlearn, tracin_influence

__all__ = ["SCRUBConfig", "scrub_unlearn", "tracin_influence"]
