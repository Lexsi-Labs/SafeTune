"""Audit S01: a zero-norm direction, an empty prompt set and an out-of-range layer used to run silently
(a no-op steer / ablate, a bare StopIteration, zero hooks). They now raise ValueError."""
import pytest
import torch

import safetune.steer as S
from safetune.interventions.steer.refusal_direction import RefusalDirectionConfig

from _steer_common import HARMFUL, HARMLESS, build

CFG = RefusalDirectionConfig(select_directions=False)


def test_identical_harmful_and_harmless_sets_raise():
    m, tok = build()
    with pytest.raises(ValueError, match="zero norm"):
        S.extract_refusal_direction(m, tok, HARMFUL, HARMFUL, CFG)


def test_zero_direction_cannot_build_a_wrapper():
    m, tok = build()
    with pytest.raises(ValueError, match="zero norm"):
        S.RefusalDirectionModel(m, torch.zeros(64), mode="steer", strength=20.0)


@pytest.mark.parametrize("which", ["harmful", "harmless"])
def test_empty_prompt_set_raises(which):
    m, tok = build()
    args = ([], HARMLESS) if which == "harmful" else (HARMFUL, [])
    with pytest.raises(ValueError, match="at least one harmful and one harmless"):
        S.extract_refusal_direction(m, tok, *args, CFG)


def test_out_of_range_layer_raises():
    m, tok = build()
    d, _, _ = S.extract_refusal_direction(m, tok, HARMFUL, HARMLESS, CFG)
    with pytest.raises(ValueError, match="out of range"):
        S.RefusalDirectionModel(m, d, mode="steer", layers=[99]).install()
    with pytest.raises(ValueError, match="pick_layer"):
        S.extract_refusal_direction(m, tok, HARMFUL, HARMLESS, RefusalDirectionConfig(pick_layer=99))


def test_empty_layers_list_raises_instead_of_installing_nothing():
    m, tok = build()
    d, _, _ = S.extract_refusal_direction(m, tok, HARMFUL, HARMLESS, CFG)
    with pytest.raises(ValueError, match="layers=\\[\\]"):
        S.RefusalDirectionModel(m, d, mode="steer", layers=[])
    assert len(S.RefusalDirectionModel(m, d, mode="steer", layers=None).install()._handles) == 6      # None still means every layer
    S.RefusalDirectionModel(m, d, layers=[1, 2]).install().remove()
