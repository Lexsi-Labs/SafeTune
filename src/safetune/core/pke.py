"""
PKE: Precision Knowledge Editing for LLM Safety -- the *locator* half.

Reference: Li et al., "Precision Knowledge Editing: Enhancing Safety in Large
Language Models", arXiv:2410.03772 (canonical repo
HydroXai/Enhancing-Safety-in-Large-Language-Models), itself a DINM-style method
(Wang et al., "Detoxifying Large Language Models via Knowledge Editing",
arXiv:2403.14472; zjunlp/EasyEdit ``dinm``).

This module implements the **localisation** stage:
:class:`ToxicNeuronLocator` ranks the per-neuron weight drift between a clean
and a toxic state dict and returns, per layer, the top-k most-drifted rows --
the neuron-weight half of DINM's ``_locate_toxic_layer``. The *edit* stage
(faithful DINM refusal cross-entropy + logit-space KL locality, optimised over
the located ``mlp.down_proj`` rows) lives in
:mod:`safetune.recover.pke` (:class:`~safetune.recover.pke.PKEGradientEditor`).

:class:`PKEEditor` below is a legacy training-free weight-copy editor retained
for backward compatibility; the faithful gradient editor in
``recover/pke.py`` supersedes it. ``max_edit_magnitude`` defaults to 1.0; pass
``None`` to disable clipping.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PKEConfig:
    """Configuration for Precision Knowledge Editing.

    Attributes:
        top_k_neurons: Number of neurons per layer to edit (ranked by
            ``|toxic - clean|`` magnitude).
        target_layers: If set, restrict editing to these layer indices.
            ``None`` means all layers.
        toxicity_weight: Reserved for future loss-based variants.  Unused
            in the current weight-arithmetic implementation.
        max_edit_magnitude: Per-element clip on the edit delta.  Defaults
            to 1.0 (was 0.1; that default was empirically too small).
            Pass ``None`` to disable clipping.
    """
    top_k_neurons: int = 50
    target_layers: Optional[List[int]] = None
    toxicity_weight: float = 0.7
    max_edit_magnitude: Optional[float] = 1.0


class ToxicNeuronLocator:
    """Identifies toxic neurons by tracking weight changes."""

    def __init__(self, config: Optional[PKEConfig] = None) -> None:
        self.config = config or PKEConfig()
        self._toxic_neurons: Dict[int, List[int]] = {}

    def locate_by_weight_change(
        self, clean_state_dict: Dict[str, Any], toxic_state_dict: Dict[str, Any],
    ) -> Dict[int, List[int]]:
        try:
            import torch
        except ImportError:
            raise ImportError("PKE requires PyTorch.")

        layer_deltas: Dict[int, List[Tuple[int, float]]] = {}
        for name in clean_state_dict:
            if name not in toxic_state_dict:
                continue
            parts = name.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is None:
                continue
            if self.config.target_layers and layer_idx not in self.config.target_layers:
                continue
            delta = (toxic_state_dict[name].float() - clean_state_dict[name].float()).abs()
            if delta.dim() >= 2:
                per_neuron = delta.sum(dim=tuple(range(1, delta.dim())))
            else:
                per_neuron = delta
            for i in range(per_neuron.shape[0]):
                layer_deltas.setdefault(layer_idx, []).append((i, per_neuron[i].item()))

        self._toxic_neurons = {}
        top_delta_per_layer: List[float] = []
        for layer_idx, neurons in layer_deltas.items():
            sorted_neurons = sorted(neurons, key=lambda x: x[1], reverse=True)
            self._toxic_neurons[layer_idx] = [
                n[0] for n in sorted_neurons[: self.config.top_k_neurons]
            ]
            if sorted_neurons:
                top_delta_per_layer.append(sorted_neurons[0][1])

        total = sum(len(v) for v in self._toxic_neurons.values())
        logger.info(
            "PKE: located %d toxic neurons across %d layers.",
            total,
            len(self._toxic_neurons),
        )
        # Sentinel: if the top per-neuron delta across every targeted layer is
        # near zero, clean and toxic state dicts are effectively identical and
        # the subsequent edit pass will be a no-op. Surface this loudly so the
        # caller can recognize the "Same as base" pattern at the source.
        if top_delta_per_layer and max(top_delta_per_layer) < 1e-8:
            logger.warning(
                "PKE: clean and toxic state dicts are effectively identical "
                "(max per-neuron delta = %.2e). The edit pass will produce no "
                "change. Check that you passed two distinct models.",
                max(top_delta_per_layer),
            )
        return self._toxic_neurons

    @property
    def toxic_neurons(self) -> Dict[int, List[int]]:
        return self._toxic_neurons


class PKEEditor:
    """Apply precision edits to identified toxic neurons."""

    def __init__(self, model: Any, toxic_neurons: Dict[int, List[int]], config: Optional[PKEConfig] = None) -> None:
        self.model = model
        self.toxic_neurons = toxic_neurons
        self.config = config or PKEConfig()

    def apply_edits(self, clean_state_dict: Dict[str, Any]) -> int:
        try:
            import torch
        except ImportError:
            raise ImportError("PKE requires PyTorch.")

        edited = 0
        current_sd = self.model.state_dict()
        for name in current_sd:
            parts = name.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is None or layer_idx not in self.toxic_neurons:
                continue
            if name not in clean_state_dict:
                continue
            target_neurons = self.toxic_neurons[layer_idx]
            tensor = current_sd[name]
            clean_tensor = clean_state_dict[name]
            if tensor.dim() >= 2:
                for neuron_idx in target_neurons:
                    if neuron_idx < tensor.shape[0]:
                        delta = clean_tensor[neuron_idx].to(tensor.dtype) - tensor[neuron_idx]
                        if self.config.max_edit_magnitude is not None:
                            cap = float(self.config.max_edit_magnitude)
                            delta = delta.clamp(-cap, cap)
                        tensor[neuron_idx] = tensor[neuron_idx] + delta
                        edited += 1
            elif tensor.dim() == 1:
                for neuron_idx in target_neurons:
                    if neuron_idx < tensor.shape[0]:
                        tensor[neuron_idx] = clean_tensor[neuron_idx].to(tensor.dtype)
                        edited += 1
        self.model.load_state_dict(current_sd)
        logger.info("PKE: edited %d neuron parameters.", edited)
        return edited
