"""RRFA (Representation Rerouting for Agentic Safety) — Steer-pillar wrapper.

Upstream
--------
RRFA is a real, verifiable project:

* Repository: https://github.com/memo-ozdincer/RRFA  (author: memo-ozdincer)
* Title: "RRFA: Representation Rerouting for Agentic Safety"

It is a research codebase (no peer-reviewed arXiv paper has been located);
the GitHub repository and author both exist and were confirmed during the
SafeTune faithfulness audit fix on 2026-05-16.

What RRFA actually is
---------------------
RRFA extends the Circuit Breakers / LoRRA framework (Zou et al., 2024) to
defend LLM *agents* against indirect prompt injection. It is a
**training-time** method: it fine-tunes **LoRA adapters** with a
circuit-breaker-style representation-rerouting loss

    L_total = alpha_benign * L_benign
            + beta_harmful * L_harmful
            + gamma_kl     * L_KL

where ``L_harmful`` pushes harmful (injection-driven) representations to be
orthogonal to the frozen baseline (``ReLU(cos_sim)`` on injection-aware
tokens), ``L_benign`` anchors benign representations to the frozen baseline
(margin-based L2 hinge), and ``L_KL`` preserves the output distribution.

Crucially, per the upstream README, at inference time RRFA "operates
automatically without explicit detection logic" — the rerouting behaviour is
*baked into the trained LoRA weights*. There is **no runtime hook, no
injection classifier, and no representation surgery applied during a forward
pass**. Once the adapter is trained the model simply behaves correctly.

Why this Steer-pillar wrapper is an honest no-op
------------------------------------------------
The SafeTune Steer pillar exposes *inference-time* interventions (forward
hooks that modify activations during generation). RRFA, being a training-time
LoRA method, has no such inference-time intervention to expose — a faithful
RRFA wrapper at generation time is, correctly, a pass-through.

Note also that the SafeTune ``core.repeng.rrfa`` module
(``InjectionDetector`` / ``RRFAWrapper``) implements a *different* mechanism
than upstream RRFA: a triplet-loss-trained runtime cosine-similarity injection
detector with whole-layer hidden-state zeroing. That runtime-detector design
is **not** the upstream RRFA algorithm and is not used here; wiring it into
generation would not make this wrapper faithful to RRFA.

Therefore ``RRFAEnsemble`` is deliberately a **documented, transparent
no-op**: it wraps a base model and passes ``generate`` / ``__call__`` straight
through, while making the situation explicit via :attr:`is_noop`,
:attr:`upstream_repo`, and a warning on construction. To actually obtain
RRFA's protection, train RRFA LoRA adapters with the upstream repository and
load the resulting adapter-merged model — at which point this (or any) plain
wrapper is sufficient because the defence lives in the weights.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

#: Verified upstream source for the RRFA method.
UPSTREAM_REPO = "https://github.com/memo-ozdincer/RRFA"


class RRFAEnsemble:
    """Base-model wrapper for RRFA (Representation Rerouting for Agentic Safety).

    RRFA is a **training-time** LoRA method (see the module docstring): its
    defence against indirect prompt injection is baked into trained adapter
    weights and there is no inference-time activation intervention to apply.

    As an inference-time Steer-pillar wrapper this class is therefore a
    deliberate, clearly-labelled **no-op pass-through**: it forwards
    ``generate`` / ``__call__`` to the wrapped model unchanged. It exists so
    that an RRFA-adapted model (one whose LoRA adapters were trained with the
    upstream repository) can be used through the same uniform Steer interface
    as the other wrappers; the wrapper itself adds no steering.

    Use :attr:`is_noop` to check this programmatically and
    :attr:`upstream_repo` for the verified upstream source.

    Parameters
    ----------
    model:
        The base causal-LM. For real RRFA protection this should already have
        RRFA-trained LoRA adapters applied/merged.
    monitor_layers, detection_threshold, detector:
        Accepted for backward-compatible call sites only. They are recorded on
        the instance but **not used** — RRFA performs no runtime detection.
    warn:
        If ``True`` (default), emit a one-time warning on construction making
        the no-op nature explicit.
    """

    #: Marks this wrapper as applying no inference-time intervention.
    is_noop: bool = True

    #: Verified upstream source for the RRFA algorithm.
    upstream_repo: str = UPSTREAM_REPO

    def __init__(
        self,
        model: Any,
        detector: Any = None,
        monitor_layers: Optional[List[int]] = None,
        detection_threshold: float = 0.6,
        warn: bool = True,
    ) -> None:
        self.model = model
        # Recorded for introspection / backward compatibility only; unused.
        self.detector = detector
        self.monitor_layers = list(monitor_layers) if monitor_layers is not None else None
        self.detection_threshold = detection_threshold

        if warn:
            logger.warning(
                "RRFAEnsemble is a no-op pass-through. RRFA "
                "(Representation Rerouting for Agentic Safety, %s) is a "
                "TRAINING-TIME LoRA method: its defence is baked into trained "
                "adapter weights and there is no inference-time activation "
                "intervention to apply. This wrapper adds no steering. Train "
                "RRFA adapters with the upstream repository to obtain the "
                "actual protection.",
                UPSTREAM_REPO,
            )
        else:
            logger.info(
                "RRFAEnsemble initialised as a documented no-op (RRFA is a "
                "training-time LoRA method; see %s).",
                UPSTREAM_REPO,
            )

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Pass through to the base model unchanged (no RRFA intervention)."""
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Pass through to the base model unchanged (no RRFA intervention)."""
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        return getattr(self.__dict__["model"], name)

    def remove_hooks(self) -> None:
        """No-op: this wrapper registers no hooks."""
        return None

    def __enter__(self) -> "RRFAEnsemble":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "RRFAEnsemble":
        """Load a base model and wrap it.

        For real RRFA protection ``path`` should point to a model whose LoRA
        adapters were trained with the upstream RRFA repository.
        """
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(path)
        return cls(model, **kwargs)
