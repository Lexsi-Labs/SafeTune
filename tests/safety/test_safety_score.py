"""Tests for SafetyScore."""
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

class TestSafetyScore:
    def test_compute(self):
        from safetune.core.eval.benchmarks import SafetyScoreAggregator
        agg = SafetyScoreAggregator()
        result = agg.compute({
            "harmfulness": 0.95,
            "over_refusal": 0.85,
            "jailbreak_resistance": 0.90,
            "capability_retention": 0.88,
            "owasp_compliance": 0.80,
        })
        assert result["passed"] is True
        assert 0 < result["aggregate_score"] <= 1.0


# ═══════════════════════════════════════════════════════════════════
# Phase 5: Guardrails
# ═══════════════════════════════════════════════════════════════════


