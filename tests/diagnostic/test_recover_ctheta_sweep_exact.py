"""sweep_ctheta_strength evaluates the model apply_ctheta would produce.

The sweep moved the weights from one strength to the next by adding ``step * delta`` to the weights
of the previous strength. Each step rounds in the target dtype, so the rounding accumulates: the
model scored at strength k was not ``target + k * delta``. In bfloat16 the weights were off by up
to 0.7% of the largest weight after five strengths, so the score reported for a strength, and the
"best" strength picked from it, belonged to a model apply_ctheta(strength=k) does not give.
"""
import copy

import pytest
import torch

from safetune.recover import apply_ctheta, sweep_ctheta_strength


class _M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(24, 24, bias=False) for _ in range(2)])


def _model(seed, dtype):
    torch.manual_seed(seed)
    return _M().to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_each_swept_model_equals_apply_ctheta_at_that_strength(dtype):
    target, positive, negative = _model(0, dtype), _model(1, dtype), _model(2, dtype)
    strengths = [0.1, 0.2, 0.3, 0.7, 1.0, 0.5]
    seen = []

    def eval_fn(model):
        seen.append({k: v.clone() for k, v in model.state_dict().items()})
        return 0.0

    before = {k: v.clone() for k, v in target.state_dict().items()}
    sweep_ctheta_strength(target, positive, negative, None, strengths, eval_fn)

    for k, got in zip(strengths, seen):
        want = apply_ctheta(copy.deepcopy(target), positive, negative, None, strength=k).state_dict()
        for name in want:
            assert torch.equal(got[name], want[name]), f"strength {k}, {name}"
    for name, w in target.state_dict().items():           # and the target is restored exactly
        assert torch.equal(w, before[name])
