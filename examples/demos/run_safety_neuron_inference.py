#!/usr/bin/env python3
"""
Example Script: SafetyNeuron Inference Pipeline

Showcases discovering dummy safety neurons and utilising them with the
Dynamic Activation Patching framework and the LLM Safeguard predictor.

This is a *mock* illustration: it uses lightweight stand-in objects instead of
a real HF model so it runs instantly with no GPU / no download. It exercises
the real ``HookedGenerationWrapper`` / ``LLMSafeguardPredictor`` control flow
(hook registration, ``setup_patching`` / ``generate`` / ``generate_safe``) — the
mocks just have to satisfy the same duck-typed interface the wrappers call:
``named_modules()``, ``register_forward_hook()``, ``forward(...)`` /
``__call__(...)`` and ``generate(...)``.

For a real run, swap ``MockModel`` for an ``AutoModelForCausalLM`` and discover
neurons with ``safetune.core.neuron_ft.discover_safety_neurons_from_model``.

Usage::

    python examples/run_safety_neuron_inference.py
"""

from safetune.core.neuron_ft import NeuronUnitScore, export_for_inference_patching
from safetune.core.runtime.inference import (
    DynamicPatchingConfig,
    HookedGenerationWrapper,
    PatchingStrategy,
    LLMSafeguardPredictor
)


class _Handle:
    """A no-op hook handle, like the one returned by torch's register_*_hook."""
    def remove(self): pass


class MockModel:
    """Minimal stand-in for a HF model exposing the hook endpoints the
    inference wrappers call. ``forward`` is a real attribute (the wrapper
    saves/restores ``base_model.forward`` during generation)."""

    def named_modules(self):
        return [("model.layers.0", self)]

    def register_forward_hook(self, hook):
        return _Handle()

    def forward(self, *args, **kwargs):
        return None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return "Safe generation."


class MockClassifier:
    """Stand-in for an sklearn classifier; column 1 is P(unsafe)."""
    def predict_proba(self, X):
        return [[0.1, 0.9]]


def main():
    print("1. Discovery phase (simulated)")
    # Normally discovered via discover_safety_neurons_from_model
    discovered_units = [
        NeuronUnitScore(unit_id="model.layers.0.12", score=0.8),
        NeuronUnitScore(unit_id="model.layers.0.45", score=0.6),
    ]
    
    print("2. Constructing Inference targets")
    # Converts to {"model.layers.0": [12, 45]}
    target_indices = export_for_inference_patching(discovered_units)
    print("Targets:", target_indices)

    print("\\n3. Initializing Dynamic Activation Patching")
    base_model = MockModel()
    guided_model = MockModel()  # E.g., a known safe LoRA instance

    config = DynamicPatchingConfig(
        target_modules=list(target_indices.keys()),
        target_indices=target_indices,
        strategy=PatchingStrategy(mode="add", scale=0.5)
    )
    
    patched_model = HookedGenerationWrapper(
        base_model=base_model,
        guided_model=guided_model,
        config=config
    )
    
    # Inference hook is automatically established during generate()
    output = patched_model.generate(["Simulate prompt generation..."])
    print("Patched generation output:", output)

    print("\\n4. Initializing LLM Safeguard Predictor")
    classifier = MockClassifier()  # Normally an SVM fit on training activations
    safeguard = LLMSafeguardPredictor(
        model=base_model,
        classifier=classifier,
        target_indices=target_indices,
        threshold=0.8
    )

    # The generate_safe wrapper aborts early if the predict_proba signals danger
    output = safeguard.generate_safe(input_ids=[1, 2, 3])
    print("Safeguard generated/aborted:", output)
    if safeguard.is_currently_unsafe:
        print("-> Generation was intercepted!")


if __name__ == "__main__":
    main()
