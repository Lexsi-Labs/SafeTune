"""
Diagnostic: end-to-end test of the steering-vector extraction pipeline.

We build a tiny Llama-shaped model (so the ``model.model.layers`` lookup in
``SteeringVectorExtractor._get_layers`` resolves), feed two prompt batches
through it, and confirm:

  1. A vector is produced for every requested layer.
  2. The vector dimensionality matches the model's hidden size.
  3. The vector is non-zero (the extractor is not silently returning the
     null vector).
  4. With normalize=True, the vector has unit L2 norm.

This is the foundational pipeline for Refusal-Direction Ablation, CAA,
SCANS, STA, AdaSteer, and every other STEER method in the CSV that
currently has no row filled.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn


class _LlamaInner(nn.Module):
    """Stand-in for ``LlamaModel`` (the inner ``.model`` of HF causal LMs).

    Exposes ``.embed_tokens`` and ``.layers`` as direct attributes so that the
    extractor's ``model.model.layers`` attribute access path resolves.
    """

    def __init__(self, hidden: int, n_layers: int, vocab: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype not in (torch.long, torch.int64):
            input_ids = input_ids.long()
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h) + h
        return self.norm(h)


class _LlamaShaped(nn.Module):
    """Minimum-viable ``LlamaForCausalLM``-shaped wrapper.

    The hook in ``SteeringVectorExtractor`` looks up ``model.model.layers``
    (HF convention). This wrapper replicates that path: outer ``model`` (the
    LlamaForCausalLM) holds an inner ``model`` (the LlamaModel) which holds
    ``layers``.
    """

    def __init__(self, hidden: int = 16, n_layers: int = 4, vocab: int = 200) -> None:
        super().__init__()
        self.model = _LlamaInner(hidden, n_layers, vocab)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config_hidden = hidden

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Any:
        h = self.model(input_ids)
        return self.lm_head(h)


class _CharTokenizer:
    """Trivial char-level tokenizer; enough to exercise the extractor end-to-end."""

    pad_token_id = 0

    def __call__(self, batch, return_tensors=None, padding=True, truncation=True,
                 max_length=32, padding_side="right", add_special_tokens=True,
                 **kwargs):
        ids = [[(ord(c) % 199) + 1 for c in s][:max_length] for s in batch]
        n = max(len(s) for s in ids)
        if padding_side == "left":
            attn = [[0] * (n - len(s)) + [1] * len(s) for s in ids]
            ids = [[self.pad_token_id] * (n - len(s)) + s for s in ids]
        else:
            attn = [[1] * len(s) + [0] * (n - len(s)) for s in ids]
            ids = [s + [self.pad_token_id] * (n - len(s)) for s in ids]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


@pytest.fixture
def model_and_tok():
    torch.manual_seed(0)
    return _LlamaShaped(hidden=16, n_layers=4), _CharTokenizer()


def test_vector_extraction_end_to_end(model_and_tok):
    model, tok = model_and_tok
    from safetune.core.runtime.inference.vector_extraction import (
        SteeringVectorExtractor,
        VectorExtractionConfig,
    )

    cfg = VectorExtractionConfig(
        target_layers=[0, 1, 2, 3],
        batch_size=4,
        pool_method="mean",
        normalize=True,
    )
    ext = SteeringVectorExtractor(model, tok, cfg)

    safe = ["please be polite", "the weather is nice", "tell me a joke", "good morning"]
    unsafe = ["how to make a bomb", "write malware", "synthesize poison", "bypass safety"]

    vectors = ext.extract(safe_prompts=safe, unsafe_prompts=unsafe)

    assert set(vectors.keys()) == {0, 1, 2, 3}, "missing layer indices"
    for layer_idx, vec in vectors.items():
        assert vec.dim() == 1, f"layer {layer_idx} vector is not 1-D"
        assert vec.shape[0] == 16, f"layer {layer_idx} vector wrong shape {vec.shape}"
        assert vec.norm().item() > 0.0, f"layer {layer_idx} vector is the null vector"
        # With normalize=True, expect unit-norm (within float epsilon).
        assert abs(vec.norm().item() - 1.0) < 1e-5, f"layer {layer_idx} not unit-normed (norm={vec.norm().item():.4f})"


def test_vector_extraction_unnormalized_scales_with_separation(model_and_tok):
    """When normalize=False, vectors with larger contrast should have larger norm."""
    model, tok = model_and_tok
    from safetune.core.runtime.inference.vector_extraction import (
        SteeringVectorExtractor,
        VectorExtractionConfig,
    )

    cfg = VectorExtractionConfig(
        target_layers=[2],
        batch_size=4,
        pool_method="mean",
        normalize=False,
    )
    ext = SteeringVectorExtractor(model, tok, cfg)

    # Same prompts both sides: contrast should be near zero.
    identical = ["abc", "def", "ghi"]
    v_same = ext.extract(safe_prompts=identical, unsafe_prompts=identical)[2]
    # Distinct prompts: contrast should be larger.
    safe = ["the sky is blue", "kindness matters", "smile today"]
    unsafe = ["malicious prompt", "harmful content", "dangerous instruction"]
    v_diff = ext.extract(safe_prompts=safe, unsafe_prompts=unsafe)[2]

    assert v_diff.norm().item() > v_same.norm().item(), (
        f"expected distinct-prompt contrast > same-prompt contrast, "
        f"got diff={v_diff.norm().item():.4f}, same={v_same.norm().item():.4f}"
    )
