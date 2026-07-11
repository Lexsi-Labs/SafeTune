"""Diagnostic: end-to-end test of CAA steering."""
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
    def __init__(self, hidden: int = 16, n_layers: int = 4, vocab: int = 200) -> None:
        super().__init__()
        self.model = _Inner(hidden, n_layers, vocab)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Any:
        return self.lm_head(self.model(input_ids))


class _Tok:
    pad_token_id = 0

    def __call__(self, batch, return_tensors=None, padding=True, truncation=True, max_length=32):
        ids = [[(ord(c) % 199) + 1 for c in s][:max_length] for s in batch]
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


def test_caa_extract_returns_per_layer_vectors(fixture):
    model, tok = fixture
    from safetune.steer import extract_caa_vectors, CAAConfig

    cfg = CAAConfig(target_layers=[0, 1, 2, 3], normalize=False)
    vecs = extract_caa_vectors(
        model,
        tok,
        positive_prompts=["good response", "helpful answer", "polite reply", "kind word"],
        negative_prompts=["rude reply", "unhelpful", "dismissive", "harsh tone"],
        config=cfg,
    )
    assert set(vecs.keys()) == {0, 1, 2, 3}
    for v in vecs.values():
        assert v.dim() == 1 and v.shape[0] == 16


def test_caa_steering_shifts_logits(fixture):
    """Applying CAA hooks should change the output logits."""
    model, tok = fixture
    from safetune.steer import extract_caa_vectors, CAAConfig, CAAModel

    inputs = tok(["test prompt please"])["input_ids"]
    with torch.no_grad():
        baseline = model(inputs).clone()

    cfg = CAAConfig(target_layers=[0, 1, 2, 3], normalize=False)
    vecs = extract_caa_vectors(
        model,
        tok,
        positive_prompts=["polite", "kind", "helpful", "honest"],
        negative_prompts=["rude", "harsh", "dismissive", "deceptive"],
        config=cfg,
    )

    with CAAModel(model, vecs, strength=3.0):
        with torch.no_grad():
            steered = model(inputs)

    delta = (steered - baseline).abs().max().item()
    assert delta > 1e-4, f"CAA hooks did not shift logits (max delta = {delta:.2e})"


def test_caa_context_manager_removes_hooks(fixture):
    """After exiting the CAAModel context, the model must produce baseline logits."""
    model, tok = fixture
    from safetune.steer import extract_caa_vectors, CAAConfig, CAAModel

    inputs = tok(["test"])["input_ids"]
    with torch.no_grad():
        before = model(inputs).clone()

    cfg = CAAConfig(target_layers=[0, 1, 2, 3])
    vecs = extract_caa_vectors(
        model, tok,
        positive_prompts=["a", "b", "c", "d"],
        negative_prompts=["e", "f", "g", "h"],
        config=cfg,
    )
    with CAAModel(model, vecs, strength=10.0):
        pass  # immediately exit

    with torch.no_grad():
        after = model(inputs)
    assert torch.allclose(before, after), "CAAModel left hooks installed after context exit"
