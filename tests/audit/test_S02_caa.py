"""Audit S02: the shared extractor's mean / max pooling includes the pad positions, so a CAA vector depends on batch size.

CAATrainer defaults to pool_method="mean" (runner/steer/_activation.py:57); the official code reads a single
position per example (nrimsky/CAA@5dabbbd, generate_vectors.py:~100, `p_activations[0, -2, :]`).
"""
import pytest
import torch

from safetune.core.runtime.inference.vector_extraction import SteeringVectorExtractor, VectorExtractionConfig

from _steer_common import HARMFUL, HARMLESS, build


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("pool", ["mean", "max"])
def test_pooled_vector_does_not_depend_on_the_batch_size(pool, side):
    m, tok = build()
    tok.padding_side = side      # the extractor pads on the left whatever the tokenizer says
    layers = list(range(6))

    def vecs(bs):
        return SteeringVectorExtractor(m, tok, VectorExtractionConfig(target_layers=layers, pool_method=pool, batch_size=bs)).extract(HARMFUL, HARMLESS)

    v1, v8 = vecs(1), vecs(8)
    scale = max(float(v1[l].abs().max()) for l in layers)
    diff = max(float((v1[l] - v8[l]).abs().max()) for l in layers)
    assert diff < 1e-3 * scale, f"batch 1 vs 8: max diff {diff:.2e} on vectors of scale {scale:.2e}"


@pytest.mark.parametrize("pool", ["mean", "max", "last_token"])
def test_unpadded_batch_is_bit_identical_to_the_plain_pool(pool):
    m, tok = build()
    ex = SteeringVectorExtractor(m, tok, VectorExtractionConfig(target_layers=[2], pool_method=pool))
    h = torch.randn(4, 7, 64)
    plain = {"mean": h.mean(dim=1), "max": h.max(dim=1).values, "last_token": h[:, -1, :]}[pool]
    assert torch.equal(ex._pool(h, torch.ones(4, 7, dtype=torch.long)), plain)
    assert torch.equal(ex._pool(h), plain)
