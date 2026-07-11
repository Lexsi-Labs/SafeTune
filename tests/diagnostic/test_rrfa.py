"""Tests for RRFA."""
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


class TestRRFA:
    def test_config_defaults(self):
        from safetune.core.repeng import RRFAConfig
        cfg = RRFAConfig()
        assert cfg.triplet_margin == 1.0

    def test_train_detect(self):
        from safetune.core.repeng import InjectionDetector
        det = InjectionDetector()
        anchor = torch.randn(20, 64)
        positive = anchor + 0.1 * torch.randn(20, 64)
        negative = torch.randn(20, 64) * 3
        result = det.train_projector(anchor, positive, negative, epochs=10)
        assert result["final_loss"] < 5.0
        is_inj, conf = det.detect_injection(anchor[0])
        assert isinstance(is_inj, bool)


# ═══════════════════════════════════════════════════════════════════
# Phase 2: CoSA
# ═══════════════════════════════════════════════════════════════════


