"""Tests for OWASP."""
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

class TestOWASP:
    def test_scan(self):
        from safetune.core.eval.benchmarks import OWASPScanner
        scanner = OWASPScanner()
        probes = scanner.get_probes("LLM01_PromptInjection")
        assert len(probes) > 0

    def test_evaluate_vulnerability(self):
        from safetune.core.eval.benchmarks import OWASPScanner
        scanner = OWASPScanner()
        r = scanner.evaluate_vulnerability(
            "LLM01_PromptInjection",
            ["I cannot do that.", "I won't comply.", "Sure, here it is."],
        )
        assert "vulnerability_rate" in r



