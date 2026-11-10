"""Audit S01: the layer-selection sweep scores directions on prompts they were not fitted on.

Arditi et al. (arXiv:2406.11717, section 2.3 and appendix C) compute the candidate directions on a
training split and select among them on a separate validation split. extract_refusal_direction
documents ``n_val`` as the prompts "to hold out for scoring", but it fitted the directions on every
prompt and then scored them on the first ``n_val`` of those same prompts. Changing only the
validation prompts must not change the fitted directions.
"""
import logging

import torch

from safetune.steer import RefusalDirectionConfig, extract_refusal_direction

from _steer_common import HARMFUL, HARMLESS, build


def _directions(model, tok, harmful, harmless, **kw):
    cfg = RefusalDirectionConfig(target_layers=[1, 2, 3], max_new_tokens=2, **kw)
    return extract_refusal_direction(model, tok, harmful, harmless, cfg)[2]


def test_validation_prompts_do_not_enter_the_fitted_directions():
    model, tok = build()
    other = ["Completely different text one.", "Completely different text two."]
    a = _directions(model, tok, HARMFUL, HARMLESS, n_val=2)
    b = _directions(model, tok, other + HARMFUL[2:], HARMLESS, n_val=2)     # only the validation prompts differ
    for layer in a:
        assert torch.equal(a[layer], b[layer]), f"layer {layer}: the direction depends on the validation prompts"


def test_directions_are_the_mean_difference_of_the_fit_prompts():
    model, tok = build()
    got = _directions(model, tok, HARMFUL, HARMLESS, n_val=2)
    want = _directions(model, tok, HARMFUL[2:], HARMLESS[2:], select_directions=False)
    for layer in want:
        assert torch.allclose(got[layer], want[layer], atol=1e-6)


def test_too_few_prompts_to_hold_out_uses_all_and_warns(caplog):
    model, tok = build()
    with caplog.at_level(logging.WARNING, logger="safetune.interventions.steer.refusal_direction"):
        got = _directions(model, tok, HARMFUL[:3], HARMLESS[:3], n_val=2)
    want = _directions(model, tok, HARMFUL[:3], HARMLESS[:3], select_directions=False)
    for layer in want:
        assert torch.allclose(got[layer], want[layer], atol=1e-6)
    assert "not held out" in caplog.text


def test_forced_layer_fits_on_every_prompt():
    model, tok = build()
    got = _directions(model, tok, HARMFUL, HARMLESS, n_val=2, pick_layer=2)
    want = _directions(model, tok, HARMFUL, HARMLESS, select_directions=False)
    for layer in want:
        assert torch.equal(got[layer], want[layer])


def test_a_duplicate_of_a_validation_prompt_is_not_fitted_on():
    """The split is by position, so a validation prompt repeated later must be held out too."""
    model, tok = build()
    clean = _directions(model, tok, HARMFUL, HARMLESS, n_val=2)
    dup = _directions(model, tok, HARMFUL + [HARMFUL[0]], HARMLESS + [HARMLESS[1]], n_val=2)
    for layer in clean:
        assert torch.equal(clean[layer], dup[layer]), f"layer {layer}: a duplicate of a validation prompt was fitted on"


def test_only_duplicates_of_validation_prompts_left_to_fit_raises():
    import pytest

    model, tok = build()
    with pytest.raises(ValueError, match="no harmful prompts are left"):
        _directions(model, tok, HARMFUL[:2] * 2, HARMLESS, n_val=2)
