"""Circuit Breakers — Representation Rerouting (RR / LoRRA) model wrapper.

Paper: "Improving Alignment and Robustness with Circuit Breakers",
Zou et al., NeurIPS 2024, arXiv:2406.04313.
Original repo: https://github.com/GraySwanAI/circuit-breakers
Key training file: ``src/lorra_circuit_breaker.py`` (``compute_loss``).

IMPORTANT — this is a TRAINING-TIME method, not an inference-time hook.
=======================================================================
Circuit Breakers fine-tunes the model (with LoRA adapters — "LoRRA") using a
two-term Representation Rerouting (RR) loss measured at a set of target
hidden layers:

  * retain loss      — keep retain-set ("benign") representations close to the
    *original* (pre-finetune) model:
        L_retain = || h_lora_retain - h_orig_retain ||_2     (mean over tokens)

  * rerouting loss   — push circuit-breaker-set ("harmful") representations
    *away* from the original model's representations, i.e. drive their cosine
    similarity to (or below) zero:
        L_cb = relu( cos_sim( h_lora_cb , h_orig_cb ) )       (mean over tokens)

  * combined, with a linear schedule over the first ~300 steps:
        c_retain = alpha * (step / T)
        c_cb     = alpha * (1 - step / T)
        L = c_retain * L_retain + c_cb * L_cb

The resulting *weights* intrinsically short-circuit harmful generations; there
is **no runtime hook** in the paper.

Why this wrapper exposes the training procedure rather than a faked hook
-----------------------------------------------------------------------
A faithful *inference-time* analogue of Circuit Breakers does not exist: the
method's entire mechanism is the learned weight update from the RR loss.
Re-casting it as a runtime projection hook (``proj > threshold``) — as an
earlier version of this file did — is a category error and an arbitrary
heuristic, not the paper. Rather than fake an inference path (the mistake
flagged for RRFA), this wrapper makes the *real* RR objective the primary,
honest API:

  * ``CircuitBreakerModel.compute_rr_loss(...)`` implements the paper's RR loss
    exactly (L2 retain term + ReLU-cosine rerouting term + linear schedule).
    Drive it inside your own LoRA fine-tuning loop to obtain a circuit-broken
    model — this is the faithful procedure.

The legacy runtime-projection hook is still reachable for backwards
compatibility via ``runtime_hook=True``, but it is explicitly **not** the
paper's method and is documented as a non-faithful heuristic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from safetune.core.repeng.circuit_breaker import (
        CircuitBreakerConfig as _CoreCBConfig,
        CircuitBreakerWrapper,
    )
    _IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    CircuitBreakerWrapper = None  # type: ignore[assignment]
    _CoreCBConfig = None  # type: ignore[assignment]
    _IMPORT_ERROR = _e


class CircuitBreakerModel:
    """Circuit Breakers (Representation Rerouting / LoRRA) wrapper.

    Faithful to Zou et al. 2024 this is a *training-time* method. The primary
    API is :meth:`compute_rr_loss`, the paper's RR loss, to be optimised inside
    a LoRA fine-tuning loop. The constructor signature is unchanged; all new
    behaviour is opt-in via keyword arguments with defaults.

    Parameters
    ----------
    model:
        The base causal-LM (the LoRA-adapted "policy" model during training).
    unsafe_directions:
        Optional per-layer directions. Only used by the legacy runtime hook.
    target_layers:
        Hidden layers at which the RR loss is measured (paper measures RR at a
        set of mid/late decoder layers). Defaults to the core config default.
    threshold:
        Legacy runtime-hook parameter (projection threshold). Unused by the RR
        training path; kept for signature compatibility.
    reroute_mode:
        Legacy runtime-hook parameter. Unused by the RR training path.
    rr_alpha:
        Overall RR loss scale ``alpha`` (paper default 5.0 / 10.0 depending on
        model). Optional kwarg, default 5.0.
    rr_schedule_steps:
        Number of steps ``T`` over which the retain/rerouting coefficients are
        linearly interpolated (paper uses 300). Optional kwarg, default 300.
    runtime_hook:
        If True, register the legacy inference-time projection hook. This is
        **not** the paper's method — see the module docstring. Default False.
    """

    def __init__(
        self,
        model: Any,
        unsafe_directions: Optional[Dict[int, Any]] = None,
        target_layers: Optional[List[int]] = None,
        threshold: float = 0.5,
        reroute_mode: str = "orthogonal",
        *,
        rr_alpha: float = 5.0,
        rr_schedule_steps: int = 300,
        runtime_hook: bool = False,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.repeng.circuit_breaker is unavailable"
            ) from _IMPORT_ERROR
        self.model = model
        cfg_kwargs: Dict[str, Any] = {
            "threshold": threshold,
            "reroute_mode": reroute_mode,
        }
        if target_layers is not None:
            cfg_kwargs["target_layers"] = list(target_layers)
        config = _CoreCBConfig(**cfg_kwargs)
        self._impl = CircuitBreakerWrapper(model=model, config=config)
        self.target_layers: List[int] = list(config.target_layers)
        self.rr_alpha = float(rr_alpha)
        self.rr_schedule_steps = int(rr_schedule_steps)
        self._runtime_hook_active = False

        # Legacy / non-faithful runtime path: only activated on explicit opt-in.
        if runtime_hook and unsafe_directions is not None:
            self._impl.set_unsafe_directions(unsafe_directions)
            self._impl.register_hooks()
            self._runtime_hook_active = True
        elif unsafe_directions is not None:
            # Stored but inert unless runtime_hook=True — keeps backwards
            # compatibility without silently faking an inference mechanism.
            self._impl.set_unsafe_directions(unsafe_directions)

    # ── faithful Representation Rerouting training loss ─────────────

    def rr_coefficients(self, step: int) -> tuple:
        """Linear RR schedule from GraySwanAI/circuit-breakers.

        ``retain_coeff = alpha * (step / T)``,
        ``cb_coeff     = alpha * (1 - step / T)``.

        Early in training the rerouting term dominates (drive harmful reps
        away); later the retain term dominates (lock in benign reps).
        """
        t = max(1, self.rr_schedule_steps)
        progress = min(1.0, max(0.0, step / t))
        retain_coeff = self.rr_alpha * progress
        cb_coeff = self.rr_alpha * (1.0 - progress)
        return retain_coeff, cb_coeff

    def compute_rr_loss(
        self,
        orig_retain_hidden: Dict[int, Any],
        lora_retain_hidden: Dict[int, Any],
        orig_cb_hidden: Dict[int, Any],
        lora_cb_hidden: Dict[int, Any],
        step: int = 0,
        retain_mask: Optional[Any] = None,
        cb_mask: Optional[Any] = None,
    ) -> Any:
        """Faithful Circuit Breakers RR loss (Zou et al. 2024).

        Mirrors ``compute_loss`` in
        ``GraySwanAI/circuit-breakers/src/lorra_circuit_breaker.py``.

        Parameters
        ----------
        orig_retain_hidden / lora_retain_hidden:
            Per-layer hidden states ``(B, S, D)`` of the *frozen original* and
            the *LoRA-adapted* model on the **retain** (benign) batch.
        orig_cb_hidden / lora_cb_hidden:
            Same, on the **circuit-breaker** (harmful) batch.
        step:
            Current global training step (drives the linear coefficient
            schedule via :meth:`rr_coefficients`).
        retain_mask / cb_mask:
            Optional attention masks ``(B, S)`` to ignore padding tokens.

        Returns
        -------
        A scalar ``torch.Tensor`` ``L = c_retain * L_retain + c_cb * L_cb``,
        differentiable w.r.t. the LoRA parameters that produced the ``lora_*``
        hidden states. Optimise it in your own fine-tuning loop.
        """
        try:
            import torch
        except ImportError:  # pragma: no cover
            raise ImportError("Circuit Breakers RR loss requires PyTorch.")

        layers = [
            l for l in self.target_layers
            if l in orig_retain_hidden and l in lora_retain_hidden
            and l in orig_cb_hidden and l in lora_cb_hidden
        ]
        if not layers:
            raise ValueError(
                "compute_rr_loss: no target layer present in all four hidden "
                f"-state dicts (target_layers={self.target_layers})."
            )

        def _stack(hd: Dict[int, Any]) -> Any:
            # (L, B, S, D) — one slice per target layer.
            return torch.stack([hd[l].float() for l in layers], dim=0)

        orig_retain = _stack(orig_retain_hidden)
        lora_retain = _stack(lora_retain_hidden)
        orig_cb = _stack(orig_cb_hidden)
        lora_cb = _stack(lora_cb_hidden)

        # ── retain loss: keep benign reps L2-close to the original model ──
        diff = lora_retain - orig_retain
        retain_norm = torch.norm(diff, dim=-1, p=2)  # (L, B, S)
        if retain_mask is not None:
            m = retain_mask.to(retain_norm.dtype)  # (B, S)
            retain_norm = retain_norm * m.unsqueeze(0)
            denom = m.sum().clamp_min(1.0) * len(layers)
            retain_loss = retain_norm.sum() / denom
        else:
            retain_loss = retain_norm.nanmean()

        # ── rerouting loss: drive harmful-rep cosine to/below zero ──
        eps = 1e-8
        norm_lora_cb = lora_cb / (lora_cb.norm(dim=-1, keepdim=True) + eps)
        norm_orig_cb = orig_cb / (orig_cb.norm(dim=-1, keepdim=True) + eps)
        cos = (norm_lora_cb * norm_orig_cb).sum(dim=-1)  # (L, B, S)
        cos = torch.relu(cos)
        if cb_mask is not None:
            m = cb_mask.to(cos.dtype)  # (B, S)
            cos = cos * m.unsqueeze(0)
            denom = m.sum().clamp_min(1.0) * len(layers)
            cb_loss = cos.sum() / denom
        else:
            cb_loss = cos.mean()

        retain_coeff, cb_coeff = self.rr_coefficients(step)
        return retain_coeff * retain_loss + cb_coeff * cb_loss

    # ── pass-through generation ────────────────────────────────────

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        return getattr(self.__dict__["model"], name)

    def remove_hooks(self) -> None:
        if hasattr(self._impl, "remove_hooks"):
            self._impl.remove_hooks()
        self._runtime_hook_active = False

    def __enter__(self) -> "CircuitBreakerModel":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "CircuitBreakerModel":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(path)
        return cls(model, **kwargs)
