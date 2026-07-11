"""
Diagnostic: end-to-end test of the Spectral Entropy Monitor.

The monitor's claim is that a hidden-state representation that has been
forced onto a low-rank subspace exhibits low spectral entropy. We test this
directly by:

1. Building a tiny model that, by construction, produces full-rank activations
   on calibration prompts.
2. Calibrating the monitor's per-layer baseline on those prompts.
3. Patching the model with a forward-hook that projects the residual stream
   onto a single direction. This is the synthetic analogue of steganographic
   encoding: information collapsed into rank-1.
4. Scanning the patched model on a new prompt. The monitor must flag the
   layer whose representation collapsed.

This also confirms the calibrate / scan / entropy_trajectory APIs work
end-to-end.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn


class _Inner(nn.Module):
    def __init__(self, hidden: int, n_layers: int, vocab: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids.long())
        for layer in self.layers:
            h = torch.nn.functional.gelu(layer(h)) + h
        return self.norm(h)


class _Wrap(nn.Module):
    def __init__(self, hidden: int = 32, n_layers: int = 6, vocab: int = 256) -> None:
        super().__init__()
        self.model = _Inner(hidden, n_layers, vocab)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Any:
        return self.lm_head(self.model(input_ids))


class _Tok:
    pad_token_id = 0

    def __call__(self, batch, return_tensors=None, padding=True, truncation=True, max_length=32):
        ids = [[(ord(c) % 255) + 1 for c in s][:max_length] for s in batch]
        n = max(len(s) for s in ids)
        attn = [[1] * len(s) + [0] * (n - len(s)) for s in ids]
        ids = [s + [self.pad_token_id] * (n - len(s)) for s in ids]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


@pytest.fixture
def fixture():
    torch.manual_seed(0)
    return _Wrap(), _Tok()


def test_calibrate_returns_per_layer_baseline(fixture):
    model, tok = fixture
    from safetune.evaluate import SpectralEntropyMonitor, SpectralMonitorConfig

    cfg = SpectralMonitorConfig(target_layers=[0, 1, 2, 3, 4, 5], batch_size=4)
    mon = SpectralEntropyMonitor(model, tok, cfg)
    baseline = mon.calibrate(
        benign_prompts=[
            "hello there how are you today",
            "the sky is bright and clear",
            "thank you for your patience",
            "good morning have a nice day",
            "the cat sat on the mat quietly",
            "books are a window into other worlds",
        ]
    )
    assert set(baseline.keys()) == {0, 1, 2, 3, 4, 5}
    for li, (mu, sigma) in baseline.items():
        assert mu > 0.0, f"layer {li} baseline mean should be positive, got {mu}"
        assert sigma > 0.0, f"layer {li} baseline std should be positive, got {sigma}"


def test_entropy_trajectory_returns_per_layer_value(fixture):
    model, tok = fixture
    from safetune.evaluate import SpectralEntropyMonitor, SpectralMonitorConfig

    mon = SpectralEntropyMonitor(model, tok, SpectralMonitorConfig(target_layers=[0, 2, 4]))
    traj = mon.entropy_trajectory("simple test prompt")
    assert set(traj.keys()) == {0, 2, 4}
    for v in traj.values():
        assert v >= 0.0


def test_scan_flags_rank1_collapse(fixture):
    """A forced rank-1 projection at one layer should produce a flagged anomaly."""
    model, tok = fixture
    from safetune.evaluate import SpectralEntropyMonitor, SpectralMonitorConfig

    mon = SpectralEntropyMonitor(
        model, tok,
        SpectralMonitorConfig(target_layers=[0, 1, 2, 3, 4, 5], z_threshold=2.0, batch_size=4),
    )
    benign = [
        "hello there how are you today",
        "the sky is bright and clear",
        "thank you for your patience",
        "good morning have a nice day",
        "the cat sat on the mat quietly",
        "books are a window into other worlds",
        "music can lift the spirit",
        "we are going to the park",
    ]
    mon.calibrate(benign)

    # Install a destructive hook on layer 3 that collapses output to rank 1.
    target_layer_idx = 3
    rank1_direction = torch.randn(32)
    rank1_direction = rank1_direction / rank1_direction.norm()

    def collapse_hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        d = rank1_direction.to(h.dtype).to(h.device)
        # Project every token onto the same 1-D subspace.
        coef = (h * d).sum(dim=-1, keepdim=True)
        h = coef * d
        return (h,) + out[1:] if isinstance(out, tuple) else h

    handle = model.model.layers[target_layer_idx].register_forward_hook(collapse_hook)
    try:
        flags = mon.scan(["this prompt will look spectrally anomalous"])
    finally:
        handle.remove()

    flagged_layers = {f[1] for f in flags}
    assert target_layer_idx in flagged_layers, (
        f"monitor did not flag rank-1 collapse at layer {target_layer_idx}; "
        f"flags = {flags}"
    )
    # The z-score for the collapsed layer should be strongly negative.
    z_for_target = min(z for (_, li, _, z) in flags if li == target_layer_idx)
    assert z_for_target < -2.0, f"collapsed-layer z-score not strongly negative: {z_for_target:.2f}"


def test_scan_without_calibration_errors(fixture):
    model, tok = fixture
    from safetune.evaluate import SpectralEntropyMonitor

    mon = SpectralEntropyMonitor(model, tok)
    with pytest.raises(RuntimeError):
        mon.scan(["any prompt"])
