"""One public API per method (ticket ST-06).

* Every harden trainer name resolves to the same class from ``safetune.harden``
  and ``safetune.runner.harden``; the ``transformers.Trainer`` subclasses are
  the ``*HFTrainer`` names, and the old spellings still work with a warning.
* Recover merge functions take ``base`` / ``aligned`` by keyword only; the old
  positional orders still work, in each function's own order, with a warning.
"""
import copy
import importlib
import importlib.util
import inspect
import warnings

import pytest
import torch
import transformers

import safetune
import safetune.harden as H
import safetune.recover as R
import safetune.runner.harden as RH

# The 15 names that used to be two different classes.
DUPLICATED = ["AsFT", "CTRAP", "ConstrainedSFT", "DOOR", "DeRTa", "Lisa", "LookAhead",
              "RepNoise", "SAP", "SEAL", "SEAM", "SPPFT", "STARDSS", "SafeGrad", "Surgery"]
# module -> method name, for the 14 whose HF class was renamed XHFTrainer
MODULES = {"safegrad": "SafeGrad", "lisa": "Lisa", "asft": "AsFT", "star_dss": "STARDSS",
           "sap": "SAP", "sppft": "SPPFT", "derta": "DeRTa", "repnoise": "RepNoise",
           "seam": "SEAM", "ctrap": "CTRAP", "seal": "SEAL",
           "constrained_sft": "ConstrainedSFT", "lookahead": "LookAhead", "surgery": "Surgery"}
TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _deprecations(record):
    return [w for w in record if issubclass(w.category, DeprecationWarning)]


# ── Harden: one class per name ────────────────────────────────────────────────

@pytest.mark.parametrize("name", DUPLICATED)
def test_harden_name_is_one_class(name):
    cls = getattr(RH, f"{name}Trainer")
    assert getattr(H, f"{name}Trainer") is cls
    assert getattr(safetune.harden, f"{name}Trainer") is cls
    ns = {}
    exec(f"from safetune.harden import {name}Trainer", ns)
    assert ns[f"{name}Trainer"] is cls


def test_every_runner_trainer_is_reachable_from_safetune_harden():
    for name in H._RUNNER_NAMES:
        assert getattr(H, name) is getattr(RH, name), name
        assert name in H.__all__ and name in dir(H)


@pytest.mark.parametrize("name", DUPLICATED)
def test_hf_trainer_behind_each_runner_trainer_stays_exported(name):
    hf = getattr(RH, f"{name}Trainer").HF_TRAINER
    assert issubclass(hf, transformers.Trainer)
    assert getattr(H, hf.__name__) is hf
    assert hf.__name__ == ("SafetyDOORTrainer" if name == "DOOR" else f"{name}HFTrainer")


@pytest.mark.parametrize("module", sorted(MODULES))
def test_submodule_alias_is_the_same_module(module):
    alias = importlib.import_module(f"safetune.harden.{module}")
    real = importlib.import_module(f"safetune.interventions.harden.{module}")
    assert alias is real
    assert getattr(alias, f"{MODULES[module]}HFTrainer") is getattr(H, f"{MODULES[module]}HFTrainer")


def test_other_pillar_submodules_are_not_loaded_twice():
    for alias, real in [("safetune.recover.merge", "safetune.interventions.recover.merge"),
                        ("safetune.evaluate.suite.evaluate",
                         "safetune.instrumentation.evaluate.suite.evaluate")]:
        assert importlib.import_module(alias) is importlib.import_module(real)
    assert importlib.util.find_spec("safetune.harden.no_such_module") is None
    with pytest.raises(ModuleNotFoundError, match="safetune.harden.no_such_module"):
        importlib.import_module("safetune.harden.no_such_module")


@pytest.mark.parametrize("module", sorted(MODULES))
def test_old_submodule_name_warns_and_returns_hf_trainer(module):
    name = MODULES[module]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ns = {}
        exec(f"from safetune.harden.{module} import {name}Trainer", ns)
    assert ns[f"{name}Trainer"] is getattr(H, f"{name}HFTrainer")
    assert len(_deprecations(rec)) == 1
    assert f"{name}HFTrainer" in str(_deprecations(rec)[0].message)


def test_runner_form_does_not_warn():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        t = H.LisaTrainer(None, None, lisa_rho=0.2)
        copy.deepcopy(t)
    assert isinstance(t, RH.LisaTrainer) and t.lisa_rho == 0.2
    assert not _deprecations(rec)


@pytest.fixture(scope="module")
def tiny_model():
    return transformers.AutoModelForCausalLM.from_pretrained(TINY_LLAMA)


def test_old_hf_call_warns_and_returns_hf_trainer(tiny_model, tmp_path):
    """safetune.harden.LisaTrainer(model=, args=, train_dataset=) was the HF
    Trainer call until 0.1.3; it keeps working for one release."""
    args = H.LisaConfig(output_dir=str(tmp_path), use_cpu=True, report_to=[])
    with pytest.warns(DeprecationWarning, match="LisaHFTrainer") as rec:
        t = H.LisaTrainer(model=tiny_model, args=args)
    assert type(t) is H.LisaHFTrainer
    assert rec[0].filename == __file__  # points at the caller's line


def test_old_positional_hf_call_warns_and_returns_hf_trainer(tiny_model, tmp_path):
    args = H.SafeGradConfig(output_dir=str(tmp_path), use_cpu=True, report_to=[])
    with pytest.warns(DeprecationWarning, match="SafeGradHFTrainer"):
        t = H.SafeGradTrainer(tiny_model, args)
    assert type(t) is H.SafeGradHFTrainer


# ── Recover: keyword-only base / aligned ──────────────────────────────────────

# (function, old positional order after the model)
MERGE_FUNCTIONS = [
    ("task_arithmetic", ("base", "aligned", "alpha")),
    ("somf_merge", ("aligned", "base", "mask_threshold", "lam", "subspace_mask")),
    ("learn_somf_mask", ("aligned", "base", "preference_data", "num_steps", "lr",
                         "temperature", "beta", "lam", "device", "seed")),
    ("apply_resta", ("base", "aligned", "alpha", "param_filter", "dare",
                     "dare_drop_rate", "dare_seed")),
    ("apply_lox", ("base", "aligned", "rank", "extrapolation_factor", "param_filter")),
    ("apply_lssf", ("base", "aligned", "alpha", "rank", "min_param_dim",
                    "skip_param_substrings", "eta", "weight_max", "subspace_basis")),
    ("apply_safemerge", ("base", "aligned", "threshold", "alpha", "merge_type",
                         "density", "only_2d")),
    ("apply_aaq", ("aligned_model_path", "base_model_path", "quantization_bits",
                   "apc_weight", "calibration_steps", "lr", "probe_texts", "top_k",
                   "simulate_quantization")),
    ("apply_safe_lora", ("aligned_state_dict_path", "aligned_state_dict",
                         "base_state_dict", "aligned_adapter_path", "base_adapter_path",
                         "alpha", "max_delta_norm")),
]


@pytest.mark.parametrize("name,legacy", MERGE_FUNCTIONS, ids=[n for n, _ in MERGE_FUNCTIONS])
def test_merge_function_references_are_keyword_only(name, legacy):
    params = list(inspect.signature(getattr(R, name)).parameters.values())
    assert params[1].kind is inspect.Parameter.KEYWORD_ONLY
    by_name = {p.name: p for p in params}
    for n in legacy:  # every old positional name still exists, keyword-only
        assert by_name[n].kind is inspect.Parameter.KEYWORD_ONLY, n


def _models(n=3, seed=0):
    torch.manual_seed(seed)
    return [torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)) for _ in range(n)]


def _weights(m):
    return torch.cat([p.detach().flatten() for p in m.parameters()])


@pytest.mark.parametrize("call", [
    # (function, keyword call, the same call in the function's old positional order)
    ("task_arithmetic", lambda f, ft, b, a: f(ft, base=b, aligned=a, alpha=0.5),
     lambda f, ft, b, a: f(ft, b, a, 0.5)),
    ("somf_merge", lambda f, ft, b, a: f(ft, base=b, aligned=a, mask_threshold=0.5),
     lambda f, ft, b, a: f(ft, a, b, 0.5)),
    ("apply_resta", lambda f, ft, b, a: f(ft, base=b, aligned=a, alpha=0.5),
     lambda f, ft, b, a: f(ft, b, a, 0.5)),
    ("apply_lssf", lambda f, ft, b, a: f(ft, base=b, aligned=a, rank=2),
     lambda f, ft, b, a: f(ft, b, a, 1.0, 2)),
], ids=lambda c: c[0])
def test_old_positional_order_warns_and_is_honoured(call):
    name, kw_call, old_call = call
    fn = getattr(R, name)
    ft, base, aligned = _models()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        expected = _weights(kw_call(fn, copy.deepcopy(ft), base, aligned))
    assert not _deprecations(rec)
    with pytest.warns(DeprecationWarning, match=f"{name}: passing") as rec:
        got = _weights(old_call(fn, copy.deepcopy(ft), base, aligned))
    assert torch.equal(got, expected)
    assert rec[0].filename == __file__


def test_somf_old_order_is_not_task_arithmetic_order():
    """The trap the ticket names: somf_merge's old order was (aligned, base).
    Swapping them changes the result, so honouring the old order matters."""
    ft, base, aligned = _models()
    right = _weights(R.somf_merge(copy.deepcopy(ft), base=base, aligned=aligned))
    swapped = _weights(R.somf_merge(copy.deepcopy(ft), base=aligned, aligned=base))
    assert not torch.equal(right, swapped)
    with pytest.warns(DeprecationWarning):
        old = _weights(R.somf_merge(copy.deepcopy(ft), aligned, base))
    assert torch.equal(old, right)


def test_target_alias_still_works():
    ft, base, aligned = _models()
    a = _weights(R.task_arithmetic(copy.deepcopy(ft), base=base, aligned=aligned))
    for kw in ("target", "finetuned", "model"):
        b = _weights(R.task_arithmetic(**{kw: copy.deepcopy(ft)}, base=base, aligned=aligned))
        assert torch.equal(a, b), kw


def test_same_argument_by_position_and_keyword_raises():
    ft, base, aligned = _models()
    with pytest.raises(TypeError, match="multiple values"), pytest.warns(DeprecationWarning):
        R.task_arithmetic(ft, base, base=base, aligned=aligned)
    with pytest.raises(TypeError), pytest.warns(DeprecationWarning):
        R.task_arithmetic(ft, base, aligned, 1.0, "one too many")


def test_learn_somf_mask_old_positional_order():
    ft, base, aligned = _models()
    with pytest.warns(DeprecationWarning, match="learn_somf_mask"):
        mask = R.learn_somf_mask(ft, aligned, base, [], 1)
    assert mask == {} or all(v.dtype == torch.bool for v in mask.values())
