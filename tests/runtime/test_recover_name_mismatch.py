"""A name-matched RECOVER method raises when no parameter name matches, instead of returning the
model untouched (issue #88).

WiseFTTrainer on a PEFT-wrapped model against a bare reference interpolated nothing: the names
carry a ``base_model.model.`` prefix, the LoRA factors have no counterpart, and the only signal
was a log line. The same skip-what-is-not-found loop sits in every weight-space RECOVER method.
"""
import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import safetune.interventions.recover as R


def _llama(seed):
    torch.manual_seed(seed)
    return LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=2, num_attention_heads=2,
                                        num_key_value_heads=2, attn_implementation="eager"))


class Wrap(torch.nn.Module):
    """Same weights, every parameter name gains an ``inner.`` prefix (as PEFT does)."""

    def __init__(self, m):
        super().__init__()
        self.inner = m

    def forward(self, *a, **k):
        return self.inner(*a, **k)


IDS = [torch.tensor([[1, 2, 3, 4, 5, 6]])]

CASES = {
    "wise_ft": lambda t, b, a: R.apply_wise_ft(t, aligned=a),
    "prepost_merge": lambda t, b, a: R.apply_prepost_merge(t, pre_model=a),
    "task_arithmetic": lambda t, b, a: R.task_arithmetic(t, base=b, aligned=a),
    "somf_merge": lambda t, b, a: R.somf_merge(t, aligned=a, base=b),
    "resta": lambda t, b, a: R.apply_resta(t, base=b, aligned=a),
    "lox": lambda t, b, a: R.apply_lox(t, base=b, aligned=a, rank=2),
    "safe_delta": lambda t, b, a: R.apply_safe_delta(t, aligned=a, unsafe=b),
    "safemerge": lambda t, b, a: R.apply_safemerge(t, base=b, aligned=a, threshold=1.0),
    "lssf": lambda t, b, a: R.apply_lssf(t, base=b, aligned=a, min_param_dim=1),
    "nlsr": lambda t, b, a: R.apply_nlsr(t, donor_state=a.state_dict()),
    "mscp": lambda t, b, a: R.apply_mscp(t, aligned_state=a.state_dict(), base_state=b.state_dict()),
    "safety_vector_restore": lambda t, b, a: R.apply_safety_vector_restore(t, aligned=a),
    "grad_selective_recover": lambda t, b, a: R.apply_grad_selective_recover(
        t, aligned=a, harmful_inputs=IDS, top_fraction=0.5),
}


def _same(m, ref_sd):
    return all(torch.equal(v, ref_sd[k]) for k, v in m.state_dict().items())


@pytest.fixture()
def models():
    base = _llama(0)
    aligned = copy.deepcopy(base)
    with torch.no_grad():
        for p in aligned.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return base, aligned


@pytest.mark.parametrize("name", CASES)
def test_no_matching_names_raises(models, name):
    base, aligned = models
    target = Wrap(copy.deepcopy(base))
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="merge_and_unload"):
        CASES[name](target, base, aligned)
    assert _same(target, before)


@pytest.mark.parametrize("name", CASES)
def test_matching_names_still_apply(models, name):
    base, aligned = models
    target = _llama(2)  # a third model, so every method has something to change
    before = copy.deepcopy(target.state_dict())
    CASES[name](target, base, aligned)
    assert not _same(target, before)


def test_wise_ft_trainer_peft_model_against_bare_reference_raises():
    """The reproduction from the issue."""
    peft = pytest.importorskip("peft")
    from safetune.runner.recover import WiseFTTrainer

    base = _llama(0)
    pm = peft.get_peft_model(
        copy.deepcopy(base),
        peft.LoraConfig(r=2, target_modules=["q_proj"], init_lora_weights=False),
    )
    with pytest.raises(ValueError, match="merge_and_unload"):
        WiseFTTrainer(model=pm, aligned_model=base, alpha=0.5).apply()

    # What the error asks for works: merge the adapter, then the names match.
    merged = pm.merge_and_unload()
    before = copy.deepcopy(merged.state_dict())
    WiseFTTrainer(model=merged, aligned_model=_llama(3), alpha=0.5).apply()
    assert not _same(merged, before)


def test_wise_ft_filter_excluding_everything_is_not_a_name_mismatch(models):
    base, aligned = models
    target = copy.deepcopy(base)
    R.apply_wise_ft(target, aligned=aligned, param_filter=lambda n, p: False)  # no raise
