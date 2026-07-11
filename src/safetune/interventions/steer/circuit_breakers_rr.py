"""
Circuit Breakers — Representation Rerouting (RR / LoRRA).

Zou, Phan, Wang, Mazeika, et al., "Improving Alignment and Robustness with
Circuit Breakers", NeurIPS 2024, arXiv:2406.04313.
Original repo: https://github.com/GraySwanAI/circuit-breakers
(training entry point: ``src/lorra_circuit_breaker.py``).

WHAT THE PAPER ACTUALLY DOES (read this before using this module)
-----------------------------------------------------------------
Circuit Breakers is a **training-time** method. It is *not* an inference-time
hook. Using a *circuit-breaker set* (harmful prompt+completion pairs) and a
*retain set* (benign data the model must keep handling normally), it fine-tunes
the model with LoRA so that harmful internal representations are *rerouted*
away from their original direction, while retain representations are left
unchanged. The trained model intrinsically short-circuits harmful generations;
no runtime intervention is needed afterwards.

The RR objective (verbatim shape from ``lorra_circuit_breaker.py``) is::

    # retain set: keep representations identical to the frozen original model
    retain_loss = || h_lora_retain - h_orig_retain ||_2          (mean)

    # circuit-breaker set: push harmful representations off their original
    # direction. inner_product is the dot of *unit-normalised* hidden states,
    # i.e. cosine similarity; relu keeps only the still-aligned part.
    cb_loss = relu( cos(h_lora_cb, h_orig_cb) )                   (masked mean)

    # linear schedule over the first ~300 steps
    progress = step / 300
    retain_coeff = alpha * progress
    cb_coeff     = alpha * (1 - progress)

    loss = retain_coeff * retain_loss + cb_coeff * cb_loss

Driving ``cos(h_lora_cb, h_orig_cb)`` down to 0 (or negative) makes the trained
harmful representation **orthogonal-or-opposite** to where the original model
put it — that is the "rerouting".

WHAT THIS MODULE PROVIDES
-------------------------
1. :meth:`CircuitBreakerRRModel.rr_loss` — a faithful, runnable implementation
   of the RR/LoRRA training objective above. This is the paper's actual
   mechanism; use it inside a fine-tuning loop (the caller supplies a frozen
   reference model / cached original hidden states).

2. An **inference-time approximation**: forward hooks that detect when the
   residual stream projects strongly onto a known harmful direction and remove
   (or orthogonalise) that component on the flagged tokens only.

   This hook is *NOT* the paper's method. It is a training-free stand-in,
   kept because callers (validation scripts, the steer registry) construct
   :class:`CircuitBreakerRRModel` with per-layer ``directions`` and expect an
   installable runtime defense. It is honestly an *approximation*: it mimics
   the *effect* of a trained circuit breaker (suppressing the harmful-aligned
   component of the representation) without the LoRA fine-tuning. The
   orthogonal-complement reroute is the closest inference analogue of what the
   RR loss achieves at convergence (cosine → 0). The faithful method is
   :meth:`rr_loss`; this hook is the convenience runtime path.

Distinct from :mod:`safetune.steer.refusal_direction`:
* refusal-direction ablation projects out one fixed direction unconditionally;
* this hook conditions on a per-position z-scored harm score, so only
  "high-harm" positions are touched (less disruptive on benign tokens).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from safetune._refusal_helpers import _get_decoder_layers

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerRRConfig:
    """Configuration for the Circuit Breakers / RR module.

    Training-time (RR/LoRRA) attributes:
        alpha: overall loss scale (``alpha`` in the paper's schedule).
        schedule_steps: number of steps over which the retain/circuit-breaker
            coefficients linearly cross over (the paper uses 300).

    Inference-time approximation attributes:
        threshold: route a token only when its projection onto the harmful
            direction exceeds this z-score relative to the per-layer mean.
        strength: scalar on the reroute (1.0 = full removal of the component).
        reroute_to: ``"zero"`` ablates the harmful component on flagged tokens;
            ``"orthogonal"`` replaces the flagged hidden state with its
            orthogonal complement w.r.t. the harmful direction (the closest
            inference analogue of the RR loss's cosine -> 0 outcome).
        mean_window: kept for backward compatibility; unused.
    """

    # inference-time approximation
    threshold: float = 1.5
    strength: float = 1.0
    reroute_to: str = "zero"
    mean_window: Optional[int] = None
    # training-time RR/LoRRA objective
    alpha: float = 5.0
    schedule_steps: int = 300


class CircuitBreakerRRModel:
    """Circuit Breakers / Representation Rerouting.

    The paper's mechanism is training-time; see :meth:`rr_loss` for a faithful
    implementation of the RR/LoRRA objective. This class additionally exposes a
    training-free inference-time *approximation* (conditional residual-stream
    rerouting) because callers construct it with per-layer harmful
    ``directions`` and expect an installable runtime hook.

    Args:
        model: HF causal LM.
        directions: ``{layer_idx: 1-D tensor (hidden,)}`` of harmful directions
            (e.g. per-layer ``mean(harmful) - mean(harmless)``). Used only by
            the inference-time approximation. May be empty / omitted when the
            object is used purely for :meth:`rr_loss` training.
        baselines: optional ``{layer_idx: (mean_proj, std_proj)}`` for z-scoring
            the projection. If absent, a running mean is used.
        config: :class:`CircuitBreakerRRConfig`.
    """

    def __init__(
        self,
        model: nn.Module,
        directions: Optional[Dict[int, torch.Tensor]] = None,
        baselines: Optional[Dict[int, tuple]] = None,
        config: Optional[CircuitBreakerRRConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or CircuitBreakerRRConfig()
        # Normalize directions to unit norm.
        self.directions: Dict[int, torch.Tensor] = {}
        for idx, vec in (directions or {}).items():
            v = vec.detach().clone()
            n = v.norm()
            if n > 1e-12:
                v = v / n
            self.directions[int(idx)] = v
        self.baselines: Dict[int, tuple] = baselines or {}
        self._handles: List[Any] = []
        self._stats: Dict[int, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Training-time RR/LoRRA objective (the paper's actual mechanism).
    # ------------------------------------------------------------------
    @staticmethod
    def rr_coeffs(step: int, alpha: float = 5.0, schedule_steps: int = 300) -> tuple:
        """Return ``(retain_coeff, circuit_breaker_coeff)`` for a training step.

        Faithful to ``lorra_circuit_breaker.py``::

            progress = step / schedule_steps          # clamped to [0, 1]
            retain_coeff = alpha * progress
            cb_coeff     = alpha * (1 - progress)

        Early in training the loss is dominated by the circuit-breaker term
        (reroute harmful representations); later it is dominated by the retain
        term (lock in benign behavior).
        """
        progress = min(max(step / max(1, schedule_steps), 0.0), 1.0)
        return alpha * progress, alpha * (1.0 - progress)

    def rr_loss(
        self,
        cb_hidden: Dict[int, torch.Tensor],
        orig_cb_hidden: Dict[int, torch.Tensor],
        retain_hidden: Dict[int, torch.Tensor],
        orig_retain_hidden: Dict[int, torch.Tensor],
        step: int = 0,
        cb_mask: Optional[torch.Tensor] = None,
        retain_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Faithful Representation Rerouting (RR/LoRRA) training loss.

        Implements the objective from GraySwanAI's ``lorra_circuit_breaker.py``.
        Call this inside a LoRA fine-tuning loop: run the (LoRA-adapted) model
        and a frozen copy of the original model on the same batch, collect
        per-layer hidden states for the circuit-breaker (harmful) and retain
        (benign) splits, and pass them here.

        Args:
            cb_hidden: ``{layer: (B, T, H)}`` hidden states of the model being
                trained, on the circuit-breaker (harmful) batch.
            orig_cb_hidden: same layers/shape from the **frozen original**
                model on the circuit-breaker batch (no grad).
            retain_hidden: hidden states of the trained model on the retain
                (benign) batch.
            orig_retain_hidden: hidden states of the frozen original model on
                the retain batch (no grad).
            step: current training step (drives the coefficient schedule).
            cb_mask: optional ``(B, T)`` attention mask for the cb batch.
            retain_mask: optional ``(B, T)`` attention mask for the retain batch.

        Returns:
            Scalar loss tensor (differentiable w.r.t. the trained model).
        """
        eps = 1e-12
        retain_coeff, cb_coeff = self.rr_coeffs(
            step, self.config.alpha, self.config.schedule_steps
        )

        # --- retain loss: keep retain representations unchanged --------------
        # retain_loss = mean || h_lora_retain - h_orig_retain ||_2
        retain_terms = []
        for layer, h in retain_hidden.items():
            if layer not in orig_retain_hidden:
                continue
            h0 = orig_retain_hidden[layer].detach().to(h.dtype)
            diff = (h - h0).float()
            norm = torch.norm(diff, dim=-1, p=2)  # (B, T)
            if retain_mask is not None:
                m = retain_mask.to(norm.dtype)
                retain_terms.append((norm * m).sum() / m.sum().clamp_min(1.0))
            else:
                retain_terms.append(norm.mean())
        retain_loss = (
            torch.stack(retain_terms).mean()
            if retain_terms
            else torch.zeros((), device=next(iter(retain_hidden.values())).device)
            if retain_hidden
            else torch.zeros(())
        )

        # --- circuit-breaker loss: reroute harmful representations -----------
        # cb_loss = masked mean of relu( cos(h_lora_cb, h_orig_cb) )
        cb_terms = []
        for layer, h in cb_hidden.items():
            if layer not in orig_cb_hidden:
                continue
            h0 = orig_cb_hidden[layer].detach().to(h.dtype)
            hf = h.float()
            h0f = h0.float()
            hn = hf / hf.norm(dim=-1, keepdim=True).clamp_min(eps)
            h0n = h0f / h0f.norm(dim=-1, keepdim=True).clamp_min(eps)
            cos = (hn * h0n).sum(dim=-1)  # (B, T) cosine similarity
            relu_cos = torch.relu(cos)
            if cb_mask is not None:
                m = cb_mask.to(relu_cos.dtype)
                cb_terms.append((relu_cos * m).sum() / m.sum().clamp_min(1.0))
            else:
                cb_terms.append(relu_cos.mean())
        cb_loss = (
            torch.stack(cb_terms).mean()
            if cb_terms
            else torch.zeros((), device=next(iter(cb_hidden.values())).device)
            if cb_hidden
            else torch.zeros(())
        )

        loss = retain_coeff * retain_loss + cb_coeff * cb_loss
        logger.debug(
            "RR loss: step=%d retain=%.4f (c=%.3f) cb=%.4f (c=%.3f) total=%.4f",
            step,
            float(retain_loss),
            retain_coeff,
            float(cb_loss),
            cb_coeff,
            float(loss),
        )
        return loss

    # ------------------------------------------------------------------
    # Inference-time approximation (training-free; NOT the paper's method).
    # ------------------------------------------------------------------
    def _make_hook(self, idx: int):
        d = self.directions[idx]
        baseline = self.baselines.get(idx)

        def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            dt = d.to(dtype=h.dtype, device=h.device)
            # per-token projection (batch, seq)
            proj = (h * dt).sum(dim=-1)
            if baseline is not None:
                mu, sigma = baseline
                z = (proj - mu) / max(sigma, 1e-6)
            else:
                running = self._stats.setdefault(idx, {"sum": 0.0, "sumsq": 0.0, "n": 0})
                cur = proj.detach().float()
                running["sum"] += float(cur.sum().item())
                running["sumsq"] += float((cur * cur).sum().item())
                running["n"] += int(cur.numel())
                mean = running["sum"] / max(1, running["n"])
                var = running["sumsq"] / max(1, running["n"]) - mean * mean
                std = max(var, 1e-6) ** 0.5
                z = (proj - mean) / std
            # Mask of tokens to reroute.
            mask = (z >= self.config.threshold).unsqueeze(-1).to(h.dtype)
            if self.config.reroute_to == "zero":
                # Subtract the harmful component on flagged tokens.
                h = h - mask * self.config.strength * (proj.unsqueeze(-1)) * dt
            elif self.config.reroute_to == "orthogonal":
                # Project flagged tokens onto the orthogonal complement of d.
                # This is the closest inference analogue of the RR loss outcome
                # (cosine of the harmful representation -> 0).
                ortho = h - (h * dt).sum(dim=-1, keepdim=True) * dt
                h = mask * ortho + (1.0 - mask) * h
            else:
                raise ValueError(f"reroute_to={self.config.reroute_to!r} not supported")
            if is_tuple:
                return (h,) + output[1:]
            return h

        return hook

    def install(self) -> "CircuitBreakerRRModel":
        """Install the inference-time approximation hooks.

        Note: this is the training-free approximation, not the paper's
        RR/LoRRA method. See :meth:`rr_loss` for the faithful objective.
        """
        self.remove()
        if not self.directions:
            logger.warning(
                "CircuitBreakerRRModel.install(): no directions provided; "
                "no hooks installed. The faithful RR mechanism is rr_loss() "
                "(training-time); the hook path needs per-layer directions."
            )
            return self
        layers = _get_decoder_layers(self.model)
        for idx in self.directions:
            if 0 <= idx < len(layers):
                self._handles.append(layers[idx].register_forward_hook(self._make_hook(idx)))
        logger.info(
            "CircuitBreakerRRModel: installed %d inference-approximation hooks "
            "(threshold=%.2f, mode=%s). This is a training-free stand-in for "
            "the RR/LoRRA training method.",
            len(self._handles),
            self.config.threshold,
            self.config.reroute_to,
        )
        return self

    def remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._stats.clear()

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        return getattr(self.__dict__["model"], name)

    def __enter__(self) -> "CircuitBreakerRRModel":
        return self.install()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()


__all__ = ["CircuitBreakerRRModel", "CircuitBreakerRRConfig"]
