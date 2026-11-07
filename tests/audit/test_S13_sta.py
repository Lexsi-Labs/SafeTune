"""Audit S13: STATrainer built `STAModel(model)` with no SAE and no vectors, so the steering was inert
(runner/steer/_activation.py); the harmful / harmless prompts it was given are unused. No CLI or config path
supplies an SAE or vectors (the CLI has no steer registry), so the trainer now takes them as arguments."""
import pytest
import torch

from safetune.runner.steer import STATrainer

from _steer_common import HARMFUL, HARMLESS, build, prompt_ids


def test_sta_trainer_raises_without_an_sae_or_vectors():
    m, tok = build()
    with pytest.raises(ValueError, match="needs an SAE"):
        STATrainer(m, tok).calibrate(HARMFUL, HARMLESS)


def test_sta_trainer_steers_with_pre_decoded_vectors():
    m, tok = build()
    ids = prompt_ids(tok, HARMFUL[0])
    with torch.no_grad():
        before = m(input_ids=ids).logits
    wrapped, _ = STATrainer(m, tok, atom_vectors={2: torch.full((64,), 0.5)}, multiplier=3.0).calibrate(HARMFUL, HARMLESS)
    with torch.no_grad():
        after = m(input_ids=ids).logits
    wrapped.remove_hooks()
    assert not torch.equal(before, after)


def test_sta_trainer_with_an_sae_and_atom_scores_builds_the_decoded_vector():
    m, tok = build()

    class FakeSAE:
        W_dec = torch.eye(64)[:8].clone()        # 8 latents -> 64-d residual
        hook_layer = 2

    scores = torch.zeros(8)
    scores[3] = 1.0
    ids = prompt_ids(tok, HARMFUL[0])
    wrapped, _ = STATrainer(m, tok, sae=FakeSAE(), atom_scores=scores, multiplier=20.0).calibrate(HARMFUL, HARMLESS)
    assert torch.equal(wrapped.layer_vectors[2], FakeSAE.W_dec[3])
    with torch.no_grad():
        steered = m(input_ids=ids).logits
    wrapped.remove_hooks()
    with torch.no_grad():
        plain = m(input_ids=ids).logits
    assert (steered - plain).abs().max() > 1e-4, "the SAE-built vector changes nothing"


@pytest.mark.parametrize("label,layers,vec,match", [
    ("NaN", {2: torch.full((64,), float("nan"))}, None, "NaN or infinite"),
    ("zero", {2: torch.zeros(64)}, None, "all zero"),
    ("wrong dimension", {2: torch.randn(10)}, None, "hidden size is 64"),
    ("out-of-range layer", {99: torch.randn(64)}, None, "out of range"),
    ("negative layer", {-1: torch.randn(64)}, None, "out of range"),
])
def test_atom_vectors_are_validated_on_construction(label, layers, vec, match):
    m, tok = build()
    with pytest.raises(ValueError, match=match):
        STATrainer(m, tok, atom_vectors=layers).calibrate(HARMFUL, HARMLESS)
    assert sum(len(l._forward_hooks) for l in m.model.layers) == 0, "a rejected wrapper left hooks behind"


def test_valid_vectors_still_steer_and_change_the_output():
    m, tok = build()
    ids = prompt_ids(tok, HARMFUL[0])
    with torch.no_grad():
        before = m(input_ids=ids).logits
    wrapped, _ = STATrainer(m, tok, atom_vectors={2: torch.full((64,), 0.5)}, multiplier=3.0).calibrate(HARMFUL, HARMLESS)
    with torch.no_grad():
        after = m(input_ids=ids).logits
    wrapped.remove_hooks()
    assert (after - before).abs().max() > 1e-3
