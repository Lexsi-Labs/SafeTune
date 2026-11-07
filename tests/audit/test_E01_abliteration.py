"""Audit E01: AbliterationAttack apply/revert bookkeeping (needs #115's refusal-direction code for the ablation itself)."""
import pytest
import torch

from safetune.evaluate import AbliterationAttack
from safetune.interventions.steer.refusal_direction import RefusalDirectionConfig

from _eval_common import HARMFUL, HARMLESS, build


def _attack():
    m, tok = build()
    a = AbliterationAttack(m, tok, RefusalDirectionConfig(select_directions=False))
    a.fit(HARMFUL, HARMLESS)
    return m, tok, a


def _n_hooks(m):
    return sum(len(mod._forward_hooks) + len(mod._forward_pre_hooks) for mod in m.modules())


def test_run_twice_then_revert_leaves_no_hook():
    m, _, a = _attack()
    a.run("runtime_ablate")
    a.run("runtime_ablate")
    a.revert()
    assert _n_hooks(m) == 0, f"{_n_hooks(m)} hooks left after revert"


def test_weight_mode_run_twice_then_revert_restores_the_original_weights():
    m, _, a = _attack()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    a.run("weight_orthogonalize")
    a.run("weight_orthogonalize")
    a.revert()
    worst = max((m.state_dict()[k] - v).abs().max().item() for k, v in before.items())
    assert worst == 0.0, f"revert left weights {worst:.3g} away from the original"


def test_weight_mode_rejects_a_strength_it_ignores():
    m, tok, a = _attack()
    before = m.model.layers[0].self_attn.o_proj.weight.clone()
    with pytest.raises(ValueError, match="no strength"):
        a.run("weight_orthogonalize", strength=0.0)
    assert torch.equal(m.model.layers[0].self_attn.o_proj.weight, before), "a rejected run still edited the weights"


def test_unknown_mode_raises_before_anything_is_reverted_or_applied():
    m, _, a = _attack()
    a.run("runtime_ablate")
    n = _n_hooks(m)
    with pytest.raises(ValueError, match="mode must be"):
        a.run("nope")
    assert _n_hooks(m) == n
    a.revert()


@pytest.mark.parametrize("mode", ["runtime_ablate", "weight_orthogonalize"])
def test_one_run_then_revert_is_bit_identical(mode):
    """The already-correct path: single apply and revert leave weights, hooks and logits exactly as before."""
    m, tok, a = _attack()
    ids = tok("PLACEHOLDER request 1: do the forbidden thing", return_tensors="pt")["input_ids"]
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        base = m(input_ids=ids).logits
    a.run(mode)
    with torch.no_grad():
        attacked = m(input_ids=ids).logits
    a.revert()
    with torch.no_grad():
        again = m(input_ids=ids).logits
    assert not torch.equal(attacked, base), "the attack changed nothing"
    assert torch.equal(again, base) and _n_hooks(m) == 0
    assert all(torch.equal(m.state_dict()[k], v) for k, v in sd.items())
