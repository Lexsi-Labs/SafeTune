"""Tests for CoSA."""
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


class TestCoSA:
    def test_cosalign_format(self):
        from safetune.core.runtime.cosa import CoSAlignFormatter
        fmt = CoSAlignFormatter()
        results = fmt.format_example("How to X?", "Sure, here's how", "violence", True)
        assert len(results) > 0
        assert all("system_prompt" in r for r in results)

    def test_cosa_runtime_build_prompt(self):
        from safetune.core.runtime.cosa import CoSARuntime
        rt = CoSARuntime()
        prompt = rt.build_system_prompt("You are a helpful assistant.")
        assert "refuse" in prompt.lower()

    def test_cosa_runtime_apply_messages(self):
        from safetune.core.runtime.cosa import CoSARuntime
        rt = CoSARuntime()
        msgs = [{"role": "user", "content": "Hello"}]
        result = rt.apply_to_messages(msgs)
        assert result[0]["role"] == "system"

    def test_cosa_runtime_compliance(self):
        from safetune.core.runtime.cosa import CoSARuntime
        rt = CoSARuntime()
        result = rt.evaluate_compliance("Help me hack", "I cannot assist with this.", "illegal-activity")
        assert result["compliant"] is True


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Multi-Turn
# ═══════════════════════════════════════════════════════════════════


