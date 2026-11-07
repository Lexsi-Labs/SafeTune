"""Audit Interpret docs: badges, citations and the readers of CircuitInfo are stated as the code behaves."""
from pathlib import Path

import safetune.instrumentation.interpret as I

ROOT = Path(I.__file__).resolve().parents[4]


def _t(p):
    return " ".join((ROOT / p).read_text().split())


def test_feature_map_badges():
    fm = (ROOT / "docs/reference/feature-map.md").read_text()
    assert "| identify_safety_neurons (weight mode) | Variant |" in fm and "| identify_safety_neurons (activation mode) | Variant |" in fm
    assert "| eap_safety_circuit (EAP / EAP-IG) | Simplified |" in fm


def test_scope_lists_the_variants_and_the_simplified_eap():
    sc = _t("docs/community/scope.md")
    assert "the 6 Variants" in sc and "identify_safety_neurons` weight mode" in sc and "Neither mode implements the cross-checkpoint metric" in sc
    assert "set-difference metric; use weight mode for a Faithful result" not in sc
    assert "`identify_safety_neurons` (weight mode), `safety_circuit_info`" not in sc and "eap_safety_circuit` is Simplified" in sc


def test_references_no_longer_cite_chen_for_the_weight_mode_and_say_node_level():
    ref = (ROOT / "docs/reference/references.md").read_text()
    row = [l for l in ref.splitlines() if l.startswith("| `identify_safety_neurons` (weight mode)")][0]
    assert "2406.14144" not in row and "aligned reference model" not in row and "SafeTune heuristic" in row
    eap = [l for l in ref.splitlines() if l.startswith("| `eap_safety_circuit`")][0]
    assert "edge resolution" not in eap and "node-level" in eap and "Simplified" in eap


def test_user_guide_pages():
    assert "Variant" in _t("docs/user-guide/interpret/safety-neurons/weight.md") and "2406.14144" in _t("docs/user-guide/interpret/safety-neurons/weight.md")
    assert '"mlp.act_fn"' not in _t("docs/user-guide/interpret/safety-neurons/activation.md").split("activation_module")[1][:60]
    eap = _t("docs/user-guide/interpret/eap/index.md")
    assert "safety-relevant edges" not in eap and "Simplified" in eap and "official default is 30" in eap
    obj = _t("docs/user-guide/interpret/circuit-info/circuit-info-object.md")
    assert "LSSF, PKE, NLSR and DeepRefusal do not read it" in obj and "unit_ids" in obj and "Protocol" in obj
