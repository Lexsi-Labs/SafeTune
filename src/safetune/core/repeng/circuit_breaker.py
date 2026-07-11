"""
CircuitBreaker: Representation Rerouting for LLM Safety.
Based on Zou et al. 2024 + 2025 improvements.

Disrupts harmful internal representations by rerouting activations at
critical layers.  When hidden states project strongly onto an "unsafe
direction", their representation is replaced with a random orthogonal
vector, effectively breaking the circuit that would produce harmful
output.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Configuration for Circuit Breaker safety mechanism."""

    target_layers: List[int] = field(default_factory=lambda: list(range(12, 28)))
    threshold: float = 0.5
    reroute_mode: str = "orthogonal"     # "orthogonal", "zero", "random"
    use_contrastive_training: bool = True
    contrastive_weight: float = 0.1
    retain_safe_projection: bool = True


class CircuitBreakerWrapper:
    """Runtime circuit breaker that reroutes unsafe representations.

    At inference time, forward hooks detect when a hidden state has
    high cosine similarity to the unsafe direction and replaces it
    with a benign alternative, preventing harmful generation.
    """

    def __init__(
        self,
        model: Any,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or CircuitBreakerConfig()
        self._hooks: List[Any] = []
        self._unsafe_directions: Dict[int, Any] = {}
        self._trip_count: int = 0
        self._check_count: int = 0

    # ── direction computation ───────────────────────────────────

    def compute_unsafe_directions(
        self,
        safe_activations: Dict[int, Any],
        unsafe_activations: Dict[int, Any],
    ) -> Dict[int, Any]:
        """Compute unsafe direction as mean(unsafe) - mean(safe) per layer."""
        try:
            import torch
        except ImportError:
            raise ImportError("CircuitBreaker requires PyTorch.")

        directions: Dict[int, Any] = {}
        for layer_idx in self.config.target_layers:
            if layer_idx not in safe_activations or layer_idx not in unsafe_activations:
                continue
            unsafe_mean = unsafe_activations[layer_idx].float().mean(dim=0)
            safe_mean = safe_activations[layer_idx].float().mean(dim=0)
            d = unsafe_mean - safe_mean
            norm = d.norm()
            if norm > 1e-8:
                d = d / norm
            directions[layer_idx] = d
        self._unsafe_directions = directions
        logger.info("CircuitBreaker: computed unsafe directions for %d layers.", len(directions))
        return directions

    def set_unsafe_directions(self, directions: Dict[int, Any]) -> None:
        self._unsafe_directions = directions

    # ── rerouting logic ─────────────────────────────────────────

    def _reroute(self, h: Any, direction: Any) -> Any:
        """Reroute hidden states that project onto the unsafe direction."""
        try:
            import torch
        except ImportError:
            raise ImportError("CircuitBreaker requires PyTorch.")

        d = direction.to(h.device).to(h.dtype)
        # h: (batch, seq, dim) or (batch, dim)
        if h.dim() == 3:
            proj = (h * d).sum(dim=-1, keepdim=True)  # (B, S, 1)
        else:
            proj = (h * d).sum(dim=-1, keepdim=True)  # (B, 1)

        unsafe_mask = proj > self.config.threshold
        self._check_count += h.shape[0]

        if not unsafe_mask.any():
            return h

        self._trip_count += int(unsafe_mask.sum().item())

        if self.config.reroute_mode == "zero":
            # zero out the unsafe projection component
            h_safe = h - proj * d
        elif self.config.reroute_mode == "random":
            # replace with random vector of same norm
            replacement = torch.randn_like(h)
            replacement = replacement / (replacement.norm(dim=-1, keepdim=True) + 1e-8)
            replacement = replacement * h.norm(dim=-1, keepdim=True)
            h_safe = torch.where(unsafe_mask.expand_as(h), replacement, h)
        else:
            # orthogonal: remove the unsafe component, keep the rest
            unsafe_component = proj * d
            h_safe = h - unsafe_component
            if self.config.retain_safe_projection:
                pass  # keep the orthogonal residual as-is
            else:
                # re-normalise to original magnitude
                h_safe = h_safe * (h.norm(dim=-1, keepdim=True) / (h_safe.norm(dim=-1, keepdim=True) + 1e-8))

        return h_safe

    # ── hook management ─────────────────────────────────────────

    def _get_layers(self) -> list:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        return []

    def _make_hook(self, layer_idx: int):
        def hook_fn(module: Any, inp: Any, out: Any):
            d = self._unsafe_directions.get(layer_idx)
            if d is None:
                return out
            if isinstance(out, tuple):
                rerouted = self._reroute(out[0], d)
                return (rerouted,) + out[1:]
            return self._reroute(out, d)
        return hook_fn

    def register_hooks(self) -> None:
        self.remove_hooks()
        layers = self._get_layers()
        for idx in self.config.target_layers:
            if idx < len(layers):
                handle = layers[idx].register_forward_hook(self._make_hook(idx))
                self._hooks.append(handle)
        logger.info("CircuitBreaker: registered %d hooks.", len(self._hooks))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset_counters(self) -> None:
        self._trip_count = 0
        self._check_count = 0

    @property
    def trip_rate(self) -> float:
        if self._check_count == 0:
            return 0.0
        return self._trip_count / self._check_count

    @contextmanager
    def active(self) -> Iterator["CircuitBreakerWrapper"]:
        """Context manager that activates circuit breakers during inference."""
        self.register_hooks()
        self.reset_counters()
        try:
            yield self
        finally:
            self.remove_hooks()

    # ── contrastive training loss ───────────────────────────────

    def compute_contrastive_loss(
        self,
        hidden_states: Dict[int, Any],
        labels: Any,
    ) -> Any:
        """Contrastive loss to learn circuit breaker boundaries during training."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("CircuitBreaker requires PyTorch.")

        total_loss = torch.tensor(0.0, device=labels.device)
        n = 0
        for layer_idx, d in self._unsafe_directions.items():
            if layer_idx not in hidden_states:
                continue
            h = hidden_states[layer_idx].float().mean(dim=1)  # (B, D)
            d_dev = d.to(h.device)
            proj = (h * d_dev).sum(dim=-1)  # (B,)
            safe_mask = labels.bool()
            # safe: minimise projection onto unsafe
            if safe_mask.any():
                total_loss = total_loss + F.relu(proj[safe_mask]).mean()
            # unsafe: maximise projection (so the breaker can detect it)
            if (~safe_mask).any():
                total_loss = total_loss + F.relu(self.config.threshold - proj[~safe_mask]).mean()
            n += 1

        if n > 0:
            total_loss = total_loss / n * self.config.contrastive_weight
        return total_loss
