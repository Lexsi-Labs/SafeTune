"""Tests for MultiTurnAttack, MTSA."""
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

class TestMultiTurnAttack:
    def test_progressive(self):
        from safetune.core.safety.multi_turn import MultiTurnAttack
        atk = MultiTurnAttack()
        msgs = atk.generate_attack("tell me how to do X")
        assert len(msgs) > 2
        assert msgs[0]["role"] == "user"

    def test_role_shift(self):
        from safetune.core.safety.multi_turn import MultiTurnAttack, MultiTurnAttackConfig
        cfg = MultiTurnAttackConfig(decomposition_strategy="role_shift")
        atk = MultiTurnAttack(cfg)
        msgs = atk.generate_attack("tell me X")
        assert any("character" in m["content"].lower() or "novel" in m["content"].lower() for m in msgs)



class TestMTSA:
    def test_generate(self):
        from safetune.core.safety.multi_turn import MTSAFormatter
        fmt = MTSAFormatter()
        results = fmt.generate_training_examples(["How to hack a server"])
        assert len(results) > 0
        assert "conversation" in results[0]


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Evaluation
# ═══════════════════════════════════════════════════════════════════


