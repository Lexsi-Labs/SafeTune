"""
Unlearn sub-pillar: selective forget-set unlearning methods.

Methods:

* :func:`scrub_unlearn` (+ :class:`SCRUBConfig`) — SCRUB teacher-student
  max-step/min-step unlearning (Kurmanji et al., NeurIPS 2023).
* :func:`rmu_unlearn` (+ :class:`RMUConfig`) — Representation Misdirection for
  Unlearning (Li et al., WMDP, 2024).
* :func:`npo_unlearn` (+ :class:`NPOConfig`) — Negative Preference Optimization
  (Zhang et al., 2024); variants npo / npo_grad_diff / npo_KL.
* :func:`gradient_ascent_unlearn` (+ :class:`GradientAscentConfig`) — GA and
  GradDiff baselines (TOFU, Maini et al., 2024). Use forget_loss="grad_diff"
  for the GradDiff variant.
* :func:`flat_unlearn` (+ :class:`FLATConfig`, :func:`flat_fdiv_loss`) — FLAT
  f-divergence loss adjustment with only forget data (Wang et al., ICLR 2025,
  arXiv:2410.11143); reference-free.
* :func:`simdpo_unlearn` (+ :class:`SimDPOUnlearnConfig`) — SimDPO-based
  safety unlearning (Chen et al., 2024); reference-free DPO.
* :func:`tracin_influence` — TracIn gradient-dot influence estimate
  (Pruthi et al., NeurIPS 2020).
"""
from .scrub import SCRUBConfig, scrub_unlearn
from .influence import tracin_influence
from .rmu import RMUConfig, rmu_unlearn
from .npo import NPOConfig, npo_unlearn, npo_forget_loss, get_batch_loss
from .gradient_ascent import GradientAscentConfig, gradient_ascent_unlearn
from .flat import FLATConfig, flat_unlearn, flat_fdiv_loss
from .simdpo import SimDPOUnlearnConfig, simdpo_unlearn, simdpo_forget_loss, make_simdpo_pairs
from .crisp import crisp_unlearn

__all__ = [
    "SCRUBConfig",
    "scrub_unlearn",
    "tracin_influence",
    "RMUConfig",
    "rmu_unlearn",
    "NPOConfig",
    "npo_unlearn",
    "npo_forget_loss",
    "get_batch_loss",
    "GradientAscentConfig",
    "gradient_ascent_unlearn",
    "FLATConfig",
    "flat_unlearn",
    "flat_fdiv_loss",
    "SimDPOUnlearnConfig",
    "simdpo_unlearn",
    "simdpo_forget_loss",
    "make_simdpo_pairs",
    "crisp_unlearn",
]