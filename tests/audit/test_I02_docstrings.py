"""Audit I02: the module docstrings say what the code is (Chen et al. 2406.14144 is a two-checkpoint RMS metric; the cosine mode is SafeTune's own)."""
import safetune.instrumentation.interpret as I
import safetune.instrumentation.interpret.eap as E
import safetune.instrumentation.interpret.safety_neurons as SN


def _flat(t):
    return " ".join(t.split())


def test_the_package_docstring_no_longer_claims_chen_et_al_use_a_refusal_cosine():
    d = _flat(I.__doc__)
    assert "matches the convention of Chen" not in d and "high cosine" not in d
    assert "Towards Understanding Safety Alignment" in d and "NeurIPS 2025" in d and "two checkpoints" in d


def test_the_module_docstring_itemises_the_declared_deviations():
    d = _flat(SN.__doc__)
    for needle in ("Variant", "one model, harmful vs. harmless prompts", "prompt tokens", "PER LAYER", "RMS of the difference",
                   "checkpoints", "Eq. 3", "Eq. 2", "no weight-space metric", "set difference"):
        assert needle in d, needle
    assert "Finding Safety Neurons" in d                        # the v1 title is mentioned as the former name


def test_readers_of_circuit_info_are_named_correctly():
    for doc in (SN.__doc__, SN.safety_circuit_info.__doc__, E.__doc__, E.eap_safety_circuit.__doc__, I.__doc__):
        d = _flat(doc)
        assert "Plug directly into LSSF" not in d and "feed LSSF / PKE" not in d and "can target" not in d
    assert "C-ΔΘ" in _flat(SN.safety_circuit_info.__doc__) and "SafetyLoRA" in _flat(E.eap_safety_circuit.__doc__)


def test_the_circuit_info_source_string_names_a_real_module():
    from safetune.instrumentation.interpret import SafetyNeuronReport
    ci = SafetyNeuronReport(per_layer={0: [(1, 0.5)]}, method="weight", target_module="mlp.down_proj").as_circuit_info()
    src = ci.layer_suggestions.metadata["source"]
    assert src == "safetune.instrumentation.interpret.safety_neurons" and __import__("importlib").import_module(src)
