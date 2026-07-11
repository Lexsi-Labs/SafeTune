"""
Safety-Layers: Localization and SPPFT.
listen0425/Safety-Layers — ICLR 2025

Localizes which transformer layers contain "safety-critical" representations
and applies Safety-Preserving Parameter Fine-Tuning (SPPFT) — freezing or
constraining updates to parameters in those safety layers.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class SafetyLayersConfig:
    """Configuration for Safety-Layers localization + SPPFT."""
    # Method for localizing safety layers: 'cosine' or 'manual'
    localization_method: str = "cosine"
    # Manually specified safety layer indices (used when method='manual')
    safety_layer_indices: List[int] = field(default_factory=list)
    # Cosine similarity threshold: layers with delta cos_sim < threshold are safety-critical
    cosine_threshold: float = 0.95
    # SPPFT mode: 'freeze' (fully freeze safety layers) or 'scale' (reduce LR)
    sppft_mode: str = "freeze"
    # LR scale factor for 'scale' mode (applied to safety layer params)
    lr_scale: float = 0.01


class SafetyLayerLocator:
    """
    Identifies which layers are safety-critical by comparing activations
    or weight changes between aligned and base models.
    """

    def __init__(self, config: Optional[SafetyLayersConfig] = None) -> None:
        self.config = config or SafetyLayersConfig()
        self._safety_layers: Set[int] = set()

    def locate_by_cosine_similarity(
        self,
        aligned_state_dict: Dict[str, Any],
        base_state_dict: Dict[str, Any],
    ) -> Set[int]:
        """
        Identify safety layers by measuring per-layer cosine similarity
        between aligned and base model weights. Layers with largest
        divergence (lowest similarity) are safety-critical.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("Safety-Layers requires PyTorch.")

        layer_sims: Dict[int, List[float]] = {}
        for name in aligned_state_dict:
            if name not in base_state_dict:
                continue
            # Detect layer index from param name (e.g., "model.layers.5.self_attn...")
            parts = name.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is None:
                continue

            a = aligned_state_dict[name].float().view(-1)
            b = base_state_dict[name].float().view(-1)
            sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            layer_sims.setdefault(layer_idx, []).append(sim)

        # Average similarity per layer
        avg_sims = {l: sum(s) / len(s) for l, s in layer_sims.items()}
        self._safety_layers = {
            l for l, sim in avg_sims.items() if sim < self.config.cosine_threshold
        }

        logger.info(
            "Safety-Layers: located %d safety-critical layers (threshold=%.3f): %s",
            len(self._safety_layers), self.config.cosine_threshold, sorted(self._safety_layers),
        )
        return self._safety_layers

    def locate(
        self,
        aligned_state_dict: Optional[Dict[str, Any]] = None,
        base_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Set[int]:
        """Locate safety layers using the configured method."""
        if self.config.localization_method == "manual":
            self._safety_layers = set(self.config.safety_layer_indices)
        elif self.config.localization_method == "cosine":
            if aligned_state_dict is None or base_state_dict is None:
                raise ValueError("Cosine localization requires aligned and base state dicts.")
            self.locate_by_cosine_similarity(aligned_state_dict, base_state_dict)
        return self._safety_layers

    @property
    def safety_layers(self) -> Set[int]:
        return self._safety_layers


class SPPFTWrapper:
    """
    Safety-Preserving Parameter Fine-Tuning: freezes or scales learning rates
    for parameters in safety-critical layers.

    Usage::

        locator = SafetyLayerLocator(config)
        safety_layers = locator.locate(aligned_sd, base_sd)
        sppft = SPPFTWrapper(model, safety_layers, config)
        sppft.apply()  # Freezes/scales safety layer params
        # ... fine-tune normally ...
        sppft.restore()  # Unfreezes after training
    """

    def __init__(
        self,
        model: Any,
        safety_layers: Set[int],
        config: Optional[SafetyLayersConfig] = None,
    ) -> None:
        self.model = model
        self.safety_layers = safety_layers
        self.config = config or SafetyLayersConfig()
        self._frozen_params: Dict[str, bool] = {}

    def _is_in_safety_layer(self, name: str) -> bool:
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1]) in self.safety_layers
        return False

    def apply(self) -> None:
        """Apply SPPFT protection to safety layer parameters."""
        mode = self.config.sppft_mode
        if mode == "scale":
            # 'scale' needs get_param_groups() wired into a custom optimizer,
            # which the standard HF Trainer path does not do — so it was a silent
            # no-op that left safety layers training unconstrained. Fall back to
            # the paper-faithful freeze (strictly safer) and say so.
            logger.warning(
                "SPPFT sppft_mode='scale' is not supported on the standard "
                "Trainer path (safety layers would train unconstrained); "
                "falling back to 'freeze'."
            )
            mode = "freeze"
        count = 0
        for name, param in self.model.named_parameters():
            if self._is_in_safety_layer(name):
                if mode == "freeze":
                    self._frozen_params[name] = param.requires_grad
                    param.requires_grad = False
                    count += 1
        logger.info("SPPFT: protected %d parameters in %d safety layers.", count, len(self.safety_layers))

    def restore(self) -> None:
        """Restore original requires_grad state."""
        for name, param in self.model.named_parameters():
            if name in self._frozen_params:
                param.requires_grad = self._frozen_params[name]
        self._frozen_params.clear()
        logger.info("SPPFT: restored all parameter states.")

    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """
        Return optimizer param groups with scaled LR for safety layers.
        Use this instead of apply() when using 'scale' mode.
        """
        safety_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if self._is_in_safety_layer(name):
                safety_params.append(param)
            else:
                other_params.append(param)

        return [
            {"params": other_params, "lr": base_lr},
            {"params": safety_params, "lr": base_lr * self.config.lr_scale},
        ]
