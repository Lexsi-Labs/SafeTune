"""
Training orchestration (internal / not yet public).

The generic SFT/DPO/PPO/GRPO orchestration backend
(``safetune.core.backend_factory``) is not implemented. Use the pillar entry
points directly instead — ``safetune.harden.*Trainer`` (train-time),
``safetune.recover.apply_*`` (weight-space), ``safetune.steer.*``
(inference-time). See docs/getting-started/taxonomy.md.
"""

from .orchestrator import build_base_config

__all__ = ["build_base_config"]
