"""
Tests for the new safety modules:
- SafeDelta
- SEAL
- SafeSwitch
- CST

(The CipherChat cases were dropped along with the legacy ``safety/attacks/``
tree — see ``verify/redteam/`` for the live stressors.)
"""

import pytest


# ─── SafeDelta ────────────────────────────────────────────────────────────────

def test_safe_delta_exports():
    from safetune.core.optim.safe_delta import SafeDeltaWrapper, SafeDeltaConfig
    cfg = SafeDeltaConfig(projection_strength=0.5)
    assert cfg.projection_strength == 0.5


def test_safe_delta_gradient_projection():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn
    from safetune.core.optim.safe_delta import SafeDeltaWrapper, SafeDeltaConfig

    model = nn.Linear(4, 4, bias=False)
    aligned_sd = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.weight.fill_(1.0)
    unsafe_sd = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.weight.fill_(0.0)  # reset to "current" state

    wrapper = SafeDeltaWrapper(model, aligned_sd, unsafe_sd)

    # Give gradient that points AGAINST safe direction
    x = torch.ones(2, 4)
    loss = model(x).sum()
    loss.backward()

    grad_before = model.weight.grad.clone()
    with wrapper.apply_safe_delta_constraint():
        pass
    grad_after = model.weight.grad.clone()

    # After projection the gradient should have changed (delta projection applied)
    assert not torch.allclose(grad_before, grad_after) or model.weight.grad.norm().item() == 0.0


# ─── SEAL ─────────────────────────────────────────────────────────────────────

def test_seal_exports():
    from safetune.core.data_compiler.seal_selector import SEALDataSelector, SEALConfig, select_safe_dataset
    cfg = SEALConfig(keep_ratio=0.6)
    assert cfg.keep_ratio == 0.6


def test_seal_filter_no_model():
    from safetune.core.data_compiler.seal_selector import SEALDataSelector, SEALConfig
    selector = SEALDataSelector(model=None, config=SEALConfig(keep_ratio=0.5))
    examples = [{"text": f"example {i}"} for i in range(10)]
    # No scores computed yet: filter warns and returns all
    result = selector.filter_dataset(examples, scores=None)
    assert result == examples


def test_seal_filter_with_scores():
    from safetune.core.data_compiler.seal_selector import SEALDataSelector, SEALConfig
    selector = SEALDataSelector(model=None, config=SEALConfig(keep_ratio=0.5))
    examples = [{"text": f"example {i}"} for i in range(10)]
    scores = [float(i) for i in range(10)]   # higher = safer
    result = selector.filter_dataset(examples, scores=scores)
    assert len(result) == 5
    # Should keep top-5 by score (examples 5-9)
    assert all(ex["text"].split()[-1] in ["5", "6", "7", "8", "9"] for ex in result)


# ─── CST ──────────────────────────────────────────────────────────────────────

def test_cst_exports():
    from safetune.core.data_compiler.cst import CSTFormatter, CSTConfig


def test_cst_format_one_example():
    from safetune.core.data_compiler.cst import CSTFormatter, CSTConfig
    formatter = CSTFormatter(CSTConfig(include_uncensored_pairs=True))
    rows = formatter.format_example("Do X?", "I can't do X.", "Sure, X is done.")
    assert len(rows) == 2
    assert rows[0]["cst_mode"] == "safe"
    assert rows[0]["chosen"] == "I can't do X."
    assert rows[1]["cst_mode"] == "uncensored"
    assert rows[1]["chosen"] == "Sure, X is done."


def test_cst_safe_only():
    from safetune.core.data_compiler.cst import CSTFormatter, CSTConfig
    formatter = CSTFormatter(CSTConfig(include_uncensored_pairs=False))
    rows = formatter.format_example("Do X?", "No.", "Yes.")
    assert len(rows) == 1
    assert rows[0]["cst_mode"] == "safe"


def test_cst_format_dataset():
    from safetune.core.data_compiler.cst import CSTFormatter
    formatter = CSTFormatter()
    examples = [
        {"prompt": f"Q{i}", "safe_response": "safe", "unsafe_response": "unsafe"}
        for i in range(5)
    ]
    rows = formatter.format_dataset(examples)
    assert len(rows) == 10   # 5 examples * 2 rows each


# ─── SafeSwitch ───────────────────────────────────────────────────────────────

def test_safeswitch_exports():
    from safetune.core.runtime.inference.safeswitch import (
        SafetyProber, SafeSwitchRunner, SafeSwitchConfig
    )


def test_safeswitch_runtime_inference_init():
    from safetune.core.runtime.inference import HookedGenerationWrapper
    from safetune.core.runtime.inference.safeswitch import SafetyProber, SafeSwitchConfig
    cfg = SafeSwitchConfig(unsafe_threshold=0.8)
    assert cfg.unsafe_threshold == 0.8
    prober = SafetyProber(hidden_size=128, layer_idx=-1)
    assert prober._clf is None  # Not trained yet
