"""Tests for UnifiedSafetyConfig."""
"""
"""
import json
import pytest
import torch
import torch.nn as nn
# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
"""Tiny model for testing hooks and representation logic."""
# ═══════════════════════════════════════════════════════════════════
# Phase 1: Representation Engineering
# ═══════════════════════════════════════════════════════════════════

class TestUnifiedSafetyConfig:
    def test_defaults(self):
        from safetune.core.safety_config import UnifiedSafetyConfig
        cfg = UnifiedSafetyConfig()
        assert cfg.evaluation.min_safety_score == 0.80

    def test_roundtrip_json(self, tmp_path):
        from safetune.core.safety_config import UnifiedSafetyConfig
        cfg = UnifiedSafetyConfig()
        path = str(tmp_path / "test_config.json")
        cfg.save_json(path)
        loaded = UnifiedSafetyConfig.from_json_file(path)
        assert loaded.evaluation.min_safety_score == cfg.evaluation.min_safety_score

    def test_from_dict(self):
        from safetune.core.safety_config import UnifiedSafetyConfig
        data = {"training": {}, "inference": {}, "evaluation": {"min_safety_score": 0.9}}
        cfg = UnifiedSafetyConfig.from_dict(data)
        assert cfg.evaluation.min_safety_score == 0.9

    def test_enabled_modules(self):
        from safetune.core.safety_config import UnifiedSafetyConfig
        cfg = UnifiedSafetyConfig()
        cfg.training.asft["enabled"] = True
        cfg.inference.adasteer["enabled"] = True
        enabled = cfg.get_enabled_modules()
        assert "asft" in enabled["training"]
        assert "adasteer" in enabled["inference"]

    def test_summary(self):
        from safetune.core.safety_config import UnifiedSafetyConfig
        cfg = UnifiedSafetyConfig()
        s = cfg.summary()
        assert "Safety Configuration Summary" in s

