"""
RepBend: Representation Bending for LLM Safety.
AIM-Intelligence/RepBend

Disrupts internal representations underlying harmful behaviour by bending
activations away from unsafe directions via a contrastive loss applied to
intermediate hidden states.  Achieves up to 95 % reduction in attack
success rates on jailbreak benchmarks while preserving utility.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RepBendConfig:
    """Configuration for Representation Bending."""

    target_layers: List[int] = field(default_factory=lambda: list(range(8, 24)))
    bending_strength: float = 0.3
    contrastive_margin: float = 1.0
    use_cosine_loss: bool = True
    safe_direction_dim: int = 128
    max_harmful_samples: int = 256


class RepBendWrapper:
    """Applies representation bending to a model during training.

    Hooks into target layers and applies a contrastive loss that
    pushes harmful representations away from the unsafe direction
    while keeping safe representations intact.
    """

    def __init__(
        self,
        model: Any,
        config: Optional[RepBendConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or RepBendConfig()
        self._hooks: List[Any] = []
        self._captured_activations: Dict[int, Any] = {}
        self._safe_direction: Optional[Any] = None

    # ── direction computation ───────────────────────────────────

    def compute_safe_direction(
        self,
        safe_activations: Dict[int, Any],
        unsafe_activations: Dict[int, Any],
    ) -> Dict[int, Any]:
        """Compute the safe direction as mean(safe) - mean(unsafe) per layer."""
        try:
            import torch
        except ImportError:
            raise ImportError("RepBend requires PyTorch.")

        directions: Dict[int, Any] = {}
        for layer_idx in self.config.target_layers:
            if layer_idx not in safe_activations or layer_idx not in unsafe_activations:
                continue
            safe_mean = safe_activations[layer_idx].float().mean(dim=0)
            unsafe_mean = unsafe_activations[layer_idx].float().mean(dim=0)
            direction = safe_mean - unsafe_mean
            norm = direction.norm()
            if norm > 1e-8:
                direction = direction / norm
            directions[layer_idx] = direction
        self._safe_direction = directions
        logger.info("RepBend: computed safe directions for %d layers.", len(directions))
        return directions

    def set_directions(self, directions: Dict[int, Any]) -> None:
        """Load precomputed safe directions."""
        self._safe_direction = directions

    # ── contrastive loss ────────────────────────────────────────

    def compute_bending_loss(
        self,
        hidden_states: Dict[int, Any],
        labels: Any,
    ) -> Any:
        """Compute contrastive representation bending loss.

        Args:
            hidden_states: dict mapping layer_idx → tensor (batch, seq, dim).
            labels: tensor (batch,) with 1 = safe, 0 = unsafe.
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("RepBend requires PyTorch.")

        if self._safe_direction is None:
            raise RuntimeError("Call compute_safe_direction() first.")

        total_loss = torch.tensor(0.0, device=labels.device)
        n_layers = 0

        for layer_idx, direction in self._safe_direction.items():
            if layer_idx not in hidden_states:
                continue
            h = hidden_states[layer_idx].float()          # (B, S, D)
            h_mean = h.mean(dim=1)                         # (B, D)
            d = direction.to(h_mean.device)

            # project onto safe direction
            proj = (h_mean * d).sum(dim=-1)                # (B,)

            safe_mask = labels.bool()
            unsafe_mask = ~safe_mask

            if self.config.use_cosine_loss:
                # safe → maximise cosine with direction
                if safe_mask.any():
                    safe_cos = F.cosine_similarity(h_mean[safe_mask], d.unsqueeze(0), dim=-1)
                    total_loss = total_loss + (1.0 - safe_cos).mean()
                # unsafe → minimise cosine (push away)
                if unsafe_mask.any():
                    unsafe_cos = F.cosine_similarity(h_mean[unsafe_mask], d.unsqueeze(0), dim=-1)
                    total_loss = total_loss + F.relu(unsafe_cos + self.config.contrastive_margin).mean()
            else:
                # margin loss
                if safe_mask.any():
                    total_loss = total_loss - proj[safe_mask].mean()
                if unsafe_mask.any():
                    total_loss = total_loss + F.relu(proj[unsafe_mask] + self.config.contrastive_margin).mean()

            n_layers += 1

        if n_layers > 0:
            total_loss = total_loss / n_layers * self.config.bending_strength

        return total_loss

    # ── hook management ─────────────────────────────────────────

    def _get_layers(self) -> list:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        return []

    def _make_hook(self, layer_idx: int):
        def hook_fn(module: Any, inp: Any, out: Any) -> None:
            if isinstance(out, tuple):
                self._captured_activations[layer_idx] = out[0].detach()
            else:
                self._captured_activations[layer_idx] = out.detach()
        return hook_fn

    def register_hooks(self) -> None:
        """Register forward hooks on target layers to capture activations."""
        self.remove_hooks()
        layers = self._get_layers()
        for idx in self.config.target_layers:
            if idx < len(layers):
                handle = layers[idx].register_forward_hook(self._make_hook(idx))
                self._hooks.append(handle)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured_activations.clear()

    @property
    def captured(self) -> Dict[int, Any]:
        return self._captured_activations

    @contextmanager
    def capture(self) -> Iterator["RepBendWrapper"]:
        """Context manager that registers hooks, yields, then removes hooks."""
        self.register_hooks()
        try:
            yield self
        finally:
            self.remove_hooks()
