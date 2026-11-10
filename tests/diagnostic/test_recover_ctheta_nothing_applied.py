"""apply_ctheta raises when it steers no parameter.

Keys are matched on the target's names and then looked up in the positive and negative models. A
key missing there, or of another shape, was skipped without a word. When that happened to every
matched key (a PEFT-wrapped target against merged references: ``...q_proj.base_layer.weight`` and
``...lora_A.default.weight`` exist in neither) the function returned the target unchanged, and the
caller went on to evaluate a "repaired" model that was the broken one. It already raises when no
key matches the layer / module filter; steering none of the matched keys is the same failure.
"""
import pytest
import torch
from peft import LoraConfig, get_peft_model

from safetune.recover import apply_ctheta, apply_ctheta_from_state_dicts


class _M(torch.nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.ModuleDict({"q_proj": torch.nn.Linear(width, width, bias=False)})
                                           for _ in range(2)])


def _model(seed, width=16):
    torch.manual_seed(seed)
    return _M(width)


def test_peft_target_with_merged_references_raises():
    target = get_peft_model(_model(0), LoraConfig(r=2, target_modules=["q_proj"]))
    with pytest.raises(ValueError, match="steered 0 of"):
        apply_ctheta(target, _model(1), _model(2), None, strength=1.0)


def test_state_dict_variant_raises_on_shape_mismatch():
    with pytest.raises(ValueError, match="steered 0 of"):
        apply_ctheta_from_state_dicts(_model(0), _model(1, width=8).state_dict(), _model(2, width=8).state_dict(), None)


def test_a_partial_match_still_applies():
    target, positive, negative = _model(0), _model(1), _model(2)
    pos_sd = {k: v for k, v in positive.state_dict().items() if "layers.0." in k}     # layer 1 is missing
    before = {k: v.clone() for k, v in target.state_dict().items()}
    apply_ctheta_from_state_dicts(target, pos_sd, negative.state_dict(), None)
    after = target.state_dict()
    assert not torch.equal(after["layers.0.q_proj.weight"], before["layers.0.q_proj.weight"])
    assert torch.equal(after["layers.1.q_proj.weight"], before["layers.1.q_proj.weight"])
