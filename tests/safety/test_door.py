"""Tests for DOOR, SafeDPO."""
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

class TestDOOR:
    def test_config(self):
        from safetune.core.optim import DOORConfig
        cfg = DOORConfig()
        assert cfg.use_weighted_refusal is True

    def test_unlearning_loss(self):
        from safetune.core.optim import DOORTrainer
        trainer = DOORTrainer()
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        loss = trainer.compute_unlearning_loss(logits, labels)
        assert loss.item() <= 0  # gradient ascent → negative



class TestSafeDPO:
    def test_generate_pairs(self):
        from safetune.core.optim import SafeDPOFormatter
        fmt = SafeDPOFormatter()
        pairs = fmt.generate_safety_pairs(
            ["How to hack?"], ["Here's how to hack..."]
        )
        assert len(pairs) == 1
        assert pairs[0]["safety_pair"] is True
        assert pairs[0]["rejected"] == "Here's how to hack..."


# ═══════════════════════════════════════════════════════════════════
# Phase 7: removed
#
# The GCG / AutoDAN / PAIR cases were dropped along with the legacy
# ``safety/attacks/`` tree — see ``verify/redteam/`` for the live stressors.
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# Phase 8: Unified Config
# ═══════════════════════════════════════════════════════════════════


