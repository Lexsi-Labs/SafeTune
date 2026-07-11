"""Tests for RepBend."""
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

class MiniModel(nn.Module):
    """Tiny model for testing hooks and representation logic."""
    def __init__(self, dim=32, n_layers=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Representation Engineering
# ═══════════════════════════════════════════════════════════════════


class TestRepBend:
    def test_config_defaults(self):
        from safetune.core.repeng import RepBendConfig
        cfg = RepBendConfig()
        assert cfg.bending_strength == 0.3
        assert len(cfg.target_layers) > 0

    def test_compute_safe_direction(self):
        from safetune.core.repeng import RepBendWrapper, RepBendConfig
        model = MiniModel(dim=16, n_layers=4)
        cfg = RepBendConfig(target_layers=[0, 1])
        wrapper = RepBendWrapper(model, cfg)
        safe = {0: torch.randn(10, 16), 1: torch.randn(10, 16)}
        unsafe = {0: torch.randn(10, 16), 1: torch.randn(10, 16)}
        dirs = wrapper.compute_safe_direction(safe, unsafe)
        assert len(dirs) == 2
        for d in dirs.values():
            assert abs(d.norm().item() - 1.0) < 1e-5

    def test_bending_loss(self):
        from safetune.core.repeng import RepBendWrapper, RepBendConfig
        model = MiniModel(dim=16, n_layers=4)
        cfg = RepBendConfig(target_layers=[0, 1], bending_strength=0.5)
        wrapper = RepBendWrapper(model, cfg)
        safe = {0: torch.randn(10, 16), 1: torch.randn(10, 16)}
        unsafe = {0: torch.randn(10, 16), 1: torch.randn(10, 16)}
        wrapper.compute_safe_direction(safe, unsafe)
        hidden = {0: torch.randn(8, 5, 16), 1: torch.randn(8, 5, 16)}
        labels = torch.tensor([1,1,1,1,0,0,0,0])
        loss = wrapper.compute_bending_loss(hidden, labels)
        assert loss.item() >= 0

    def test_hook_capture(self):
        from safetune.core.repeng import RepBendWrapper, RepBendConfig
        model = MiniModel(dim=16, n_layers=4)
        cfg = RepBendConfig(target_layers=[0, 1, 2])
        wrapper = RepBendWrapper(model, cfg)
        with wrapper.capture() as w:
            _ = model(torch.randn(2, 16))
            assert len(w.captured) > 0



