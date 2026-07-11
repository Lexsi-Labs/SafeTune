"""
SafetyNeuronFT: phase-1 neuron/feature-centric safety utilities.

This module provides a lightweight, CircuitKIT-optional implementation for:
- safety-neuron discovery via activation contrast placeholders,
- causal impact validation hooks,
- targeted intervention specs (gating/scaling/masking).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class NeuronUnitScore:
    """Discovered candidate neuron with safety relevance score."""
    unit_id: str
    score: float
    layer: Optional[int] = None
    module: Optional[str] = None


@dataclass
class SafetyNeuronIntervention:
    """Targeted intervention payload for downstream patching/adapters."""
    mode: str = "gating"  # gating | scaling | masking
    units: List[NeuronUnitScore] = field(default_factory=list)
    scale: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyNeuronFTConfig:
    """Configuration for SafetyNeuronFT track."""
    discovery_method: str = "activation_contrast"
    top_k: int = 64
    causal_validation: bool = True
    intervention_mode: str = "gating"
    intervention_scale: float = 0.0
    use_circuit_hints: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_safety_neuronft_config(**kwargs: Any) -> SafetyNeuronFTConfig:
    """Build config from a dict-like source (e.g., YAML safety.methods.params)."""
    return SafetyNeuronFTConfig(**kwargs)


def discover_safety_neurons(
    harmful_activations: Dict[str, float],
    benign_activations: Dict[str, float],
    config: SafetyNeuronFTConfig,
) -> List[NeuronUnitScore]:
    """
    Phase-1 discovery by activation contrast.
    The method computes absolute delta between harmful and benign means.
    """
    scores: List[NeuronUnitScore] = []
    unit_ids = set(harmful_activations.keys()) | set(benign_activations.keys())
    for unit_id in unit_ids:
        h = float(harmful_activations.get(unit_id, 0.0))
        b = float(benign_activations.get(unit_id, 0.0))
        delta = abs(h - b)
        scores.append(NeuronUnitScore(unit_id=unit_id, score=delta))
    scores.sort(key=lambda x: x.score, reverse=True)
    return scores[: max(0, int(config.top_k))]


def validate_neuron_causality(
    units: List[NeuronUnitScore],
    ablation_score_fn: Optional[Callable[[List[str]], float]] = None,
) -> float:
    """Return causal validation score in [0,1] using optional ablation callback."""
    if not units:
        return 0.0
    if ablation_score_fn is None:
        # No runtime ablation available in phase 1; use normalized proxy.
        return min(1.0, max(0.0, float(sum(u.score for u in units) / len(units))))
    try:
        score = float(ablation_score_fn([u.unit_id for u in units]))
    except Exception:
        return 0.0
    return min(1.0, max(0.0, score))


def build_intervention(
    units: List[NeuronUnitScore],
    config: SafetyNeuronFTConfig,
    validation_score: Optional[float] = None,
) -> SafetyNeuronIntervention:
    """Create a standardized intervention payload for adapters/patch tracks."""
    metadata: Dict[str, Any] = {"discovery_method": config.discovery_method}
    if validation_score is not None:
        metadata["validation_score"] = validation_score
    return SafetyNeuronIntervention(
        mode=config.intervention_mode,
        units=units,
        scale=config.intervention_scale,
        metadata=metadata,
    )


def export_for_inference_patching(units: List[NeuronUnitScore]) -> Dict[str, List[int]]:
    """
    Helper to convert a list of discovered NeuronUnitScore instances into the
    module_name -> [indices] mapping expected by DynamicPatchingConfig and LLMSafeguardPredictor.
    """
    mapping: Dict[str, List[int]] = {}
    for u in units:
        # Expected format of unit_id from ActivationCapture: "module.name.idx" or similar.
        # But if we don't have explicit indices, the discovery might just provide module names.
        # Safest fallback is assuming unit_id splits into module_name and an integer index.
        parts = u.unit_id.rsplit(".", 1)
        if len(parts) == 2 and parts[1].isdigit():
            mod, idx = parts[0], int(parts[1])
            mapping.setdefault(mod, []).append(idx)
        else:
            # If unit_ids are just module names, map them to an empty list (meaning all indices)
            mapping.setdefault(u.unit_id, [])
    # Sort indices for deterministic behavior
    for mod in mapping:
        mapping[mod].sort()
    return mapping


# ---------------------------------------------------------------------------
# Real PyTorch activation capture via forward hooks
# ---------------------------------------------------------------------------

class ActivationCapture:
    """Context manager that captures mean activations from named modules.

    Usage::

        model = AutoModelForCausalLM.from_pretrained(...)
        with ActivationCapture(model, module_names=["model.layers.0.mlp"]) as cap:
            model(input_ids)
        print(cap.activations)  # {"model.layers.0.mlp": 0.42}

    Args:
        model: Any ``nn.Module`` (e.g. HuggingFace causal LM).
        module_names: List of fully-qualified submodule names to capture.
            An empty list or ``None`` captures *all* named modules.
        reduction: ``"mean"`` (default) stores the scalar mean absolute
            activation; ``"max"`` stores the max absolute activation.
    """

    def __init__(
        self,
        model: Any,
        module_names: Optional[List[str]] = None,
        reduction: str = "mean",
    ) -> None:
        self._model = model
        self._target_names: Optional[set] = set(module_names) if module_names else None
        self._reduction = reduction
        self._handles: List[Any] = []
        self.activations: Dict[str, float] = {}

    def __enter__(self) -> "ActivationCapture":
        try:
            import torch.nn as nn
        except ImportError:
            return self

        def _make_hook(name: str):
            def _hook(module, input, output):
                try:
                    import torch
                    # output can be a tensor or a tuple; grab first tensor
                    tensor = output if isinstance(output, torch.Tensor) else (
                        output[0] if isinstance(output, (tuple, list)) and len(output) > 0 else None
                    )
                    if tensor is None:
                        return
                    flat = tensor.detach().float().abs().view(-1)
                    if self._reduction == "max":
                        val = float(flat.max().item())
                    else:
                        val = float(flat.mean().item())
                    # Accumulate: keep running mean across multiple forward passes
                    if name in self.activations:
                        self.activations[name] = (self.activations[name] + val) / 2.0
                    else:
                        self.activations[name] = val
                except Exception:
                    pass
            return _hook

        # Enumerate all submodules; include root ("") only when it's the only module
        # so that bare nn.Linear / leaf models are captured correctly.
        all_named = list(self._model.named_modules())
        only_root = len(all_named) == 1  # no sub-modules at all

        for name, module in all_named:
            effective_name = name if name else "<root>"
            if not only_root and not name:  # skip root when real sub-modules exist
                continue
            if self._target_names and effective_name not in self._target_names and name not in self._target_names:
                continue
            h = module.register_forward_hook(_make_hook(effective_name))
            self._handles.append(h)


        return self

    def __exit__(self, *args: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def collect_activations(
    model: Any,
    inputs: Any,
    module_names: Optional[List[str]] = None,
    reduction: str = "mean",
) -> Dict[str, float]:
    """Run a single forward pass and return mean activations per module.

    Args:
        model: HuggingFace or bare ``nn.Module``.
        inputs: Anything accepted by ``model.forward()`` —
            e.g. ``{"input_ids": tensor}`` or a tuple.
        module_names: Submodule names to capture (``None`` = all).
        reduction: ``"mean"`` or ``"max"``.

    Returns:
        Dict mapping submodule name → scalar activation statistic.
    """
    try:
        import torch
    except ImportError:
        return {}

    with ActivationCapture(model, module_names=module_names, reduction=reduction) as cap:
        with torch.no_grad():
            if isinstance(inputs, dict):
                model(**inputs)
            elif isinstance(inputs, (tuple, list)):
                model(*inputs)
            else:
                model(inputs)

    return cap.activations


def discover_safety_neurons_from_model(
    model: Any,
    harmful_inputs: Any,
    benign_inputs: Any,
    config: SafetyNeuronFTConfig,
    module_names: Optional[List[str]] = None,
) -> List[NeuronUnitScore]:
    """High-level helper: run discovery using real model forward passes.

    Captures activations on harmful and benign inputs, then delegates to
    ``discover_safety_neurons()`` for scoring.

    Args:
        model: HuggingFace causal LM or any ``nn.Module``.
        harmful_inputs: Batch passed to ``model.forward()`` for harmful examples.
        benign_inputs: Batch passed to ``model.forward()`` for benign examples.
        config: ``SafetyNeuronFTConfig`` controlling top-k and method.
        module_names: Submodule names to capture. ``None`` captures all.

    Returns:
        Top-k ``NeuronUnitScore`` instances sorted by safety relevance.
    """
    harmful_acts = collect_activations(model, harmful_inputs, module_names=module_names)
    benign_acts = collect_activations(model, benign_inputs, module_names=module_names)
    return discover_safety_neurons(harmful_acts, benign_acts, config)
