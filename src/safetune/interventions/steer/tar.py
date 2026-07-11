"""TAR (Tamper-Resistant Safeguards) model wrapper.

Paper — "Tamper-Resistant Safeguards for Open-Weight LLMs", Tamirisa et al.,
ICLR 2025, arXiv:2408.00761. Repo: https://github.com/rishub-tamirisa/tamper-resistance

IMPORTANT — TAR IS A TRAINING-TIME METHOD; THIS WRAPPER APPLIES NOTHING AT GENERATION TIME.
======================================================================================
TAR is a meta-learning *defense applied during safety training*. Its outer loop trains the
model's weights so that even after ``K`` simulated adversarial fine-tuning steps the model
still refuses harmful queries:

    clone params  ->  run K inner adversarial SGD steps  ->  evaluate a tamper-resistance
    (safety) loss on the *tampered* params  ->  accumulate that into a meta-gradient on the
    original weights  ->  outer SGD step.

The protection lives entirely in the *trained weights*. There is **no runtime hook, no
activation edit, and no logit edit** in the paper — once trained, the model is just run
normally. Consequently a faithful *inference-time* analogue of TAR genuinely does not exist:
there is nothing to intervene on at generation time without inventing a heuristic that is
not the paper's method.

The previous version of ``TARModel`` was therefore a silent **pass-through**: it held a
``TARWrapper`` as ``self._impl`` but registered no hooks and applied no intervention, so
``generate``/``__call__`` went straight to the base model. Presented under the Steer
(inference-time intervention) pillar that is a category mismatch and the audit rated it 🟡.

This module follows the honest stance taken for the Circuit Breakers fix: rather than dress
up a no-op as a steering method, ``TARModel`` is now explicit that it performs **no
inference-time intervention** and routes callers to the real, training-time TAR objective.

WHERE THE REAL METHOD LIVES
---------------------------
The faithful, training-time TAR objective is implemented in
``safetune.harden.tar.tar_outer_loss`` (the Harden pillar) — that function builds the
first-order meta-gradient (retain loss + lambda_tar * g_TR) for one batch triple and is the
function to use to actually obtain a tamper-resistant model. ``TARModel.train_step`` below
is a thin convenience forwarder to it so the training objective is reachable from a
consistent wrapper API; it does not change ``harden/`` in any way.

``TARModel`` still exists so the Steer pillar exposes a consistent model API (``generate``,
``__call__``, ``from_pretrained``) across methods, and so that a model *already* hardened
with TAR can be carried through Steer-style code unchanged — but callers must understand
that the safety here came from training, not from this wrapper.
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    from safetune.core.repeng.tar import (
        TARConfig as _CoreTARConfig,
        TARWrapper,
    )
    _IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    TARWrapper = None  # type: ignore[assignment]
    _CoreTARConfig = None  # type: ignore[assignment]
    _IMPORT_ERROR = _e


class TARModel:
    """Wrapper exposing a :class:`TARWrapper` alongside a base model.

    TAR is a **training-time meta-learning** defense (Tamirisa et al., ICLR 2025).
    It has **no inference-time intervention**: this wrapper does *not* register hooks,
    does *not* edit activations, and does *not* edit logits. ``generate``/``__call__``
    are deliberate, documented pass-throughs to the base model — the tamper resistance
    comes from how the weights were *trained*, not from anything this wrapper does.

    To actually train a tamper-resistant model use the faithful training-time objective
    :func:`safetune.harden.tar.tar_outer_loss` (exposed here via :meth:`train_step`).

    Args:
        model: the base causal-LM (ideally one already hardened with TAR training).
        inner_steps: ``K``, the number of simulated adversarial SGD steps the TAR
            outer loop unrolls. Used only by the training-time objective.
        inner_lr: learning rate of the simulated adversary's inner SGD steps.
        target_modules: optional name substrings restricting which parameters are
            made tamper-resistant; ``None`` means all parameters.
        warn: if ``True`` (default), emit a one-time warning that this wrapper applies
            no inference-time intervention, so the no-op behaviour is never silent.
    """

    #: TAR is training-time only — this wrapper performs no generation-time intervention.
    is_inference_intervention: bool = False

    def __init__(
        self,
        model: Any,
        inner_steps: int = 4,
        inner_lr: float = 2e-5,
        target_modules: Optional[List[str]] = None,
        warn: bool = True,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.repeng.tar is unavailable"
            ) from _IMPORT_ERROR
        self.model = model
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.target_modules = list(target_modules) if target_modules is not None else None
        config = _CoreTARConfig(
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            target_modules=self.target_modules,
        )
        self._impl = TARWrapper(model=model, config=config)
        if warn:
            import warnings

            warnings.warn(
                "TARModel applies NO inference-time intervention: TAR is a "
                "training-time meta-learning defense (Tamirisa et al., ICLR 2025). "
                "generate()/__call__() pass straight through to the base model. "
                "Use TARModel.train_step(...) / safetune.harden.tar.tar_outer_loss "
                "to actually obtain a tamper-resistant model.",
                stacklevel=2,
            )

    # ── training-time objective (the real TAR method) ──────────────

    def train_step(
        self,
        retain_batch: Any,
        harm_batch: Any,
        safety_batch: Any,
        task_loss_fn: Any,
        safety_loss_fn: Optional[Any] = None,
    ) -> Any:
        """Compute the faithful TAR outer-loop loss for one batch triple.

        This is a thin forwarder to :func:`safetune.harden.tar.tar_outer_loss` — the
        real, training-time TAR objective. It builds the first-order meta-gradient
        (retain loss + ``lambda_tar`` * g_TR); call ``.backward()`` on the returned
        scalar inside a training loop to perform a TAR outer update.

        ``inner_steps`` / ``inner_lr`` / ``target_modules`` passed to this wrapper's
        constructor are forwarded via a :class:`safetune.harden.tar.TARConfig`.

        Note: this does **not** make ``generate`` do anything — it is the training
        path. See the module docstring.
        """
        from safetune.harden.tar import TARConfig as _HardenTARConfig, tar_outer_loss

        cfg = _HardenTARConfig(
            inner_steps=self.inner_steps,
            inner_lr=self.inner_lr,
        )
        return tar_outer_loss(
            self.model,
            retain_batch=retain_batch,
            harm_batch=harm_batch,
            safety_batch=safety_batch,
            task_loss_fn=task_loss_fn,
            config=cfg,
            safety_loss_fn=safety_loss_fn,
        )

    # ── inference path (deliberate, documented pass-through) ───────

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Pass straight through to the base model.

        TAR applies no inference-time intervention; this is a documented no-op
        wrapper around ``self.model.generate`` (see the module docstring).
        """
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Pass straight through to the base model (no inference-time intervention)."""
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        return getattr(self.__dict__["model"], name)

    def remove_hooks(self) -> None:
        """No-op: TAR registers no forward hooks (training-time method)."""
        return None

    def __enter__(self) -> "TARModel":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "TARModel":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(path)
        return cls(model, **kwargs)
