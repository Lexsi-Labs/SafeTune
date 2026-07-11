"""
Diagnostic: vLLM-Lens accelerated steering backend.

These tests do not require vLLM or vllm-lens to be installed. They cover:

1. ``SteeringSpec`` dataclass shape and defaults.
2. ``VllmSteeredBackend.is_available()`` reports correctly.
3. ``make_backend("vllm-lens", ...)`` falls back gracefully when the optional
   deps are missing (degraded to plain vLLM, or HF if even vLLM is absent).
4. Constructing ``VllmSteeredBackend`` without any steering vectors raises.

The actual steered-inference path is exercised by integration tests in a
GPU-enabled environment with vllm-lens installed; see
``tests/diagnostic/test_real_model_smoke.py`` for the HF-hooks-based
counterpart that runs unconditionally on GPU 1.
"""
from __future__ import annotations

import pytest
import torch


def test_steering_spec_defaults():
    from safetune.core.eval.pipeline.backends.vllm_lens import SteeringSpec

    v = torch.randn(16)
    spec = SteeringSpec(vector=v, layer_indices=[10])
    assert spec.scale == 0.5
    assert spec.norm_match is True
    assert spec.position_indices is None
    assert list(spec.layer_indices) == [10]


def test_vllm_steered_backend_is_available_returns_bool():
    from safetune.core.eval.pipeline.backends.vllm_lens import VllmSteeredBackend

    assert isinstance(VllmSteeredBackend.is_available(), bool)


def test_vllm_steered_backend_requires_vectors():
    from safetune.core.eval.pipeline.backends.vllm_lens import VllmSteeredBackend

    with pytest.raises(ValueError):
        VllmSteeredBackend(model="meta-llama/Llama-3.2-1B-Instruct", steering_vectors=[])


def test_make_backend_vllm_lens_fallback_when_missing():
    """If vllm-lens is not installed, ``make_backend("vllm-lens", ...)`` must
    degrade to one of the available backends without raising. The Generator
    runs but without steering; that path is logged as a warning."""
    from safetune.core.eval.pipeline.backends import make_backend
    from safetune.core.eval.pipeline.backends.vllm_lens import VllmSteeredBackend, SteeringSpec

    if VllmSteeredBackend.is_available():
        pytest.skip("vllm-lens is installed; fallback path not exercised here.")

    v = torch.randn(16)
    spec = SteeringSpec(vector=v, layer_indices=[5])
    backend = make_backend(
        "vllm-lens",
        model="meta-llama/Llama-3.2-1B-Instruct",
        steering_vectors=[spec],
    )
    # Should be a non-vllm-lens backend. It still has a .generate method.
    assert hasattr(backend, "generate")
    assert not isinstance(backend, VllmSteeredBackend)


def test_make_backend_unknown_after_addition():
    """Adding the vllm-lens alias must not break the unknown-backend path."""
    from safetune.core.eval.pipeline.backends import make_backend

    with pytest.raises(ValueError):
        make_backend("nonexistent-backend", model="x")
