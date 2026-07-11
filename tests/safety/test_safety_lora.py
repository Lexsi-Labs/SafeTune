"""Tests for SafetyLoRA config."""
import pytest


def test_safety_lora_without_circuit():
    from safetune.core.safety_lora import build_safety_lora_config
    config = build_safety_lora_config(
        lora_r=8,
        lora_target_modules=["q_proj", "v_proj"],
        circuit_guided=False,
    )
    assert config.circuit_guided is False
    assert config.resolve_target_modules() == ["q_proj", "v_proj"]


def test_safety_lora_with_circuit_info():
    from safetune.core.safety_lora import SafetyLoRAConfig
    from safetune.core.circuit_kit import CircuitInfo, LayerModuleSuggestions
    info = CircuitInfo(layer_suggestions=LayerModuleSuggestions(target_modules=["v_proj", "o_proj"]))
    config = SafetyLoRAConfig(lora_target_modules=["q_proj"], circuit_guided=True, circuit_info=info)
    assert config.resolve_target_modules() == ["v_proj", "o_proj"]
