"""Red-team / stressor surface for the Verify pillar.

Verify has two functions — *red-teaming* (stressing a model) and *evaluation*
(judging / benchmarking). This package is the red-team half.

Only the two stressors that the faithfulness audit verified are kept here:

- ``AbliterationAttack`` — a **weight-space drift condition** (refusal-direction
  ablation), not a prompt attack. It is the in-house drift stressor for C-ΔΘ.
- ``BoNAttack`` — Best-of-N input-space jailbreak.

The 12 other in-house attack reimplementations (GCG, PAIR, AutoDAN, TAP,
ArtPrompt, FlipAttack, ReNeLLM, CodeChameleon, Cipher, Steg, MTSA, Virus) were
removed: the audit found them broken or only approximate. Source any further
red-team / tamper stress from maintained external harnesses — TamperBench
(arXiv:2602.06911) and ``cthetha-eval`` — instead of reimplementing them.
"""

from .abliteration import AbliterationAttack
from .bon import BoNAttack, BoNConfig, AUGMENTATIONS

__all__ = [
    "AbliterationAttack",
    "BoNAttack",
    "BoNConfig",
    "AUGMENTATIONS",
]
