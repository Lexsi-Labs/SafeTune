"""Tests for CircuitBreaker."""
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


class TestCircuitBreaker:
    def test_config_defaults(self):
        from safetune.core.repeng import CircuitBreakerConfig
        cfg = CircuitBreakerConfig()
        assert cfg.threshold == 0.5

    def test_reroute_orthogonal(self):
        from safetune.core.repeng import CircuitBreakerWrapper, CircuitBreakerConfig
        model = MiniModel(dim=16, n_layers=4)
        cfg = CircuitBreakerConfig(target_layers=[0], threshold=0.0, reroute_mode="orthogonal")
        wrapper = CircuitBreakerWrapper(model, cfg)
        d = torch.randn(16)
        d = d / d.norm()
        wrapper.set_unsafe_directions({0: d})
        h = d.unsqueeze(0) * 2  # strongly aligned with unsafe
        result = wrapper._reroute(h, d)
        proj_on_unsafe = (result * d).sum().item()
        assert abs(proj_on_unsafe) < 1e-3  # should be orthogonal

    def test_contrastive_loss(self):
        from safetune.core.repeng import CircuitBreakerWrapper, CircuitBreakerConfig
        model = MiniModel(dim=16, n_layers=4)
        cfg = CircuitBreakerConfig(target_layers=[0])
        wrapper = CircuitBreakerWrapper(model, cfg)
        d = torch.randn(16); d = d / d.norm()
        wrapper.set_unsafe_directions({0: d})
        h = {0: torch.randn(4, 5, 16)}
        labels = torch.tensor([1,1,0,0])
        loss = wrapper.compute_contrastive_loss(h, labels)
        assert loss.item() >= 0

    def test_context_manager(self):
        from safetune.core.repeng import CircuitBreakerWrapper, CircuitBreakerConfig
        model = MiniModel(dim=16, n_layers=4)
        cfg = CircuitBreakerConfig(target_layers=[0, 1])
        wrapper = CircuitBreakerWrapper(model, cfg)
        d = torch.randn(16); d = d / d.norm()
        wrapper.set_unsafe_directions({0: d, 1: d})
        with wrapper.active() as w:
            _ = model(torch.randn(2, 16))
        assert w.trip_rate >= 0



