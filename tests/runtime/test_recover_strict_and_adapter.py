"""RECOVER name matching, part two (issue #88): the two paths that still skipped silently, the
opt-in ``strict=`` for partial coverage, and ``merge_adapter=True`` for a PEFT model.

``test_recover_name_mismatch.py`` covers "no name matches"; this file covers the rest.
"""
import copy
import logging

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import safetune.interventions.recover as R
from tests.runtime.test_recover_name_mismatch import CASES, Wrap, _llama, _same


@pytest.fixture()
def models():
    base = _llama(0)
    aligned = copy.deepcopy(base)
    with torch.no_grad():
        for p in aligned.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return base, aligned


# ---- the two paths that used to skip silently ------------------------------------------------

PROBE = torch.randint(0, 32, (2, 6))


def _safereact(t, ref, **kw):
    return R.apply_safereact(t, reference_model=ref, top_k_neurons=8, train_lora=False,
                             probe_inputs=PROBE, **kw)


def test_safereact_fallback_raises_when_no_name_matches():
    ref, post = _llama(0), _llama(1)
    target = Wrap(copy.deepcopy(post))
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="merge_and_unload"):
        _safereact(target, ref)
    assert _same(target, before)


def test_safereact_fallback_still_applies_when_names_match():
    ref, post = _llama(0), _llama(1)
    target = copy.deepcopy(post)
    before = copy.deepcopy(target.state_dict())
    _safereact(target, ref)
    assert not _same(target, before)


def test_safereact_strict_is_a_noop_on_a_normal_payload():
    # the payload names only the targeted weights; a full match of its own names passes strict
    ref, post = _llama(0), _llama(1)
    _safereact(copy.deepcopy(post), ref, strict=True)


def _donor_map(model, n=4):
    return {k: {str(i): 0.5 for i in range(n)} for k, p in model.named_parameters() if p.dim() == 2}


def test_nlsr_donor_map_raises_when_no_name_matches():
    target = Wrap(_llama(1))
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="merge_and_unload"):
        R.apply_nlsr(target, donor_map=_donor_map(_llama(1)))
    assert _same(target, before)


def test_nlsr_without_any_donor_raises():
    target = _llama(1)
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="no parameter name"):
        R.apply_nlsr(target)
    assert _same(target, before)


def test_nlsr_donor_map_still_applies_and_strict_checks_the_map():
    target = _llama(1)
    dm = _donor_map(target)
    before = copy.deepcopy(target.state_dict())
    R.apply_nlsr(target, donor_map=dm, strict=True)
    assert not _same(target, before)

    # a map name that is not a model parameter: applied by default, refused by strict
    dm["no.such.weight"] = {"0": 1.0}
    target2 = _llama(1)
    before2 = copy.deepcopy(target2.state_dict())
    R.apply_nlsr(target2, donor_map=dm)
    assert not _same(target2, before2)
    target3 = _llama(1)
    before3 = copy.deepcopy(target3.state_dict())
    with pytest.raises(ValueError, match="strict=True"):
        R.apply_nlsr(target3, donor_map=dm, strict=True)
    assert _same(target3, before3)


# ---- strict= for partial coverage ------------------------------------------------------------

def _with_extra_param():
    t = _llama(2)
    t.register_parameter("extra_w", torch.nn.Parameter(torch.zeros(2, 2)))  # no counterpart
    return t


@pytest.mark.parametrize("name", CASES)
def test_partial_overlap_applies_by_default_and_strict_refuses(models, name, caplog):
    base, aligned = models
    target = _with_extra_param()
    before = copy.deepcopy(target.state_dict())
    with caplog.at_level(logging.WARNING):
        CASES[name](target, base, aligned)  # default: unchanged behaviour
    assert not _same(target, before)
    assert any("extra_w" in r.getMessage() and "left unchanged" in r.getMessage() for r in caplog.records)

    target = _with_extra_param()
    before = copy.deepcopy(target.state_dict())
    kw = {"strict": True}
    with pytest.raises(ValueError, match="strict=True"):
        _STRICT[name](target, base, aligned, kw)
    assert _same(target, before)  # raised before any edit


_STRICT = {
    "wise_ft": lambda t, b, a, kw: R.apply_wise_ft(t, aligned=a, **kw),
    "prepost_merge": lambda t, b, a, kw: R.apply_prepost_merge(t, pre_model=a, **kw),
    "task_arithmetic": lambda t, b, a, kw: R.task_arithmetic(t, base=b, aligned=a, **kw),
    "somf_merge": lambda t, b, a, kw: R.somf_merge(t, aligned=a, base=b, **kw),
    "resta": lambda t, b, a, kw: R.apply_resta(t, base=b, aligned=a, **kw),
    "lox": lambda t, b, a, kw: R.apply_lox(t, base=b, aligned=a, rank=2, **kw),
    "safe_delta": lambda t, b, a, kw: R.apply_safe_delta(t, aligned=a, unsafe=b, **kw),
    "safemerge": lambda t, b, a, kw: R.apply_safemerge(t, base=b, aligned=a, threshold=1.0, **kw),
    "lssf": lambda t, b, a, kw: R.apply_lssf(t, base=b, aligned=a, min_param_dim=1, **kw),
    "nlsr": lambda t, b, a, kw: R.apply_nlsr(t, donor_state=a.state_dict(), **kw),
    "mscp": lambda t, b, a, kw: R.apply_mscp(t, aligned_state=a.state_dict(), base_state=b.state_dict(), **kw),
    "safety_vector_restore": lambda t, b, a, kw: R.apply_safety_vector_restore(t, aligned=a, **kw),
    "grad_selective_recover": lambda t, b, a, kw: R.apply_grad_selective_recover(
        t, aligned=a, harmful_inputs=[torch.tensor([[1, 2, 3, 4, 5, 6]])], top_fraction=0.5, **kw),
}


def test_strict_passes_when_every_name_matches(models):
    base, aligned = models
    target = _llama(2)
    R.apply_wise_ft(target, aligned=aligned, strict=True)
    R.apply_lox(target, base=base, aligned=aligned, rank=2, strict=True)


def test_trainer_strict_reaches_the_method(models):
    from safetune.runner.recover import WiseFTTrainer

    _, aligned = models
    with pytest.raises(ValueError, match="strict=True"):
        WiseFTTrainer(model=_with_extra_param(), aligned_model=aligned, strict=True).apply()
    WiseFTTrainer(model=_with_extra_param(), aligned_model=aligned).apply()


# ---- merge_adapter=True for a PEFT model -----------------------------------------------------

@pytest.fixture()
def lora():
    peft = pytest.importorskip("peft")
    base = _llama(0)
    pm = peft.get_peft_model(
        copy.deepcopy(base),
        peft.LoraConfig(r=2, target_modules=["q_proj", "v_proj"], init_lora_weights=False),
    )
    merged = copy.deepcopy(pm).merge_and_unload()
    return base, pm, merged


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("fn", ["wise_ft", "prepost_merge"])
def test_merge_adapter_scales_the_lora_update_by_one_minus_alpha(lora, fn, alpha):
    base, pm, merged = lora
    before = copy.deepcopy(pm.state_dict())
    call = {
        "wise_ft": lambda: R.apply_wise_ft(pm, aligned=base, alpha=alpha, merge_adapter=True),
        "prepost_merge": lambda: R.apply_prepost_merge(pm, pre_model=base, alpha=alpha, merge_adapter=True),
    }[fn]
    out = call()
    assert _same(pm, before)  # the PEFT model is untouched
    assert not hasattr(out, "peft_config") and set(out.state_dict()) == set(base.state_dict())
    for k, v in out.state_dict().items():  # base + (1 - alpha) * lora update
        want = base.state_dict()[k] + (1 - alpha) * (merged.state_dict()[k] - base.state_dict()[k])
        torch.testing.assert_close(v, want, atol=1e-6, rtol=1e-5)


def test_merge_adapter_accepts_a_state_dict_reference(lora):
    base, pm, merged = lora
    out = R.apply_wise_ft(pm, aligned=base.state_dict(), alpha=1.0, merge_adapter=True)
    assert _same(out, base.state_dict())


def test_merge_adapter_on_a_model_without_an_adapter_raises():
    with pytest.raises(ValueError, match="PEFT model"):
        R.apply_wise_ft(_llama(1), aligned=_llama(0), merge_adapter=True)


def test_trainer_merge_adapter(lora):
    from safetune.runner.recover import PrePostMergeTrainer, WiseFTTrainer

    base, pm, merged = lora
    out = WiseFTTrainer(model=pm, aligned_model=base, alpha=0.5, merge_adapter=True).apply()
    assert not hasattr(out, "peft_config")
    out = PrePostMergeTrainer(model=pm, pre_model=base, alpha=0.5, merge_adapter=True).apply()
    assert not hasattr(out, "peft_config")
