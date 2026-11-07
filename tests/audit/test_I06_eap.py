"""Audit I06: eap_safety_circuit vs Syed et al. 2310.10348 (EAP) / Hanna et al. 2403.17806 (EAP-IG) and hannamw/EAP-IG@e24bafd (src/eap/attribute.py).
Random tiny Llama with the tiny test tokenizer (skipped if it is not cached); placeholder prompts."""
import copy

import pytest
import torch

import safetune.instrumentation.interpret.eap as E
from safetune.instrumentation.interpret import EAPSafetyCircuitConfig, eap_safety_circuit

from _interpret_common import HARMFUL, HARMLESS, build


def _cfg(**kw):
    kw.setdefault("method", "eap")
    kw.setdefault("granularity", "block")
    kw.setdefault("max_seq_len", 32)
    kw.setdefault("batch_size", len(HARMFUL))
    return EAPSafetyCircuitConfig(**kw)


def test_scores_do_not_depend_on_the_batch_size():
    m, tok = build()
    base = E._run_eap(m, tok, "cpu", HARMFUL, HARMLESS, _cfg(batch_size=6))
    for bs in (3, 2, 1, 4):                                   # 4: a partial last batch
        got = E._run_eap(m, tok, "cpu", HARMFUL, HARMLESS, _cfg(batch_size=bs))
        assert all(abs(got[n] - base[n]) <= 1e-5 + 1e-4 * abs(base[n]) for n in base), (bs, {n: round(got[n] / base[n], 3) for n in list(base)[:3]})


def test_scores_are_per_pair_means():
    m, tok = build()
    one = E._run_eap(m, tok, "cpu", HARMFUL[:1], HARMLESS[:1], _cfg(batch_size=1))
    two = E._run_eap(m, tok, "cpu", HARMFUL[:2] * 3, HARMLESS[:2] * 3, _cfg(batch_size=3))     # the same two pairs, three times each
    both = E._run_eap(m, tok, "cpu", HARMFUL[:2], HARMLESS[:2], _cfg(batch_size=2))
    assert all(abs(two[n] - both[n]) <= 1e-5 + 1e-4 * abs(both[n]) for n in both)
    assert set(one) == set(both)


def test_hooks_are_removed_and_the_loaded_model_is_left_as_found():
    m, tok = build()
    m.train()
    n_hooks = lambda: sum(len(x._forward_hooks) + len(x._forward_pre_hooks) for x in m.modules())
    before = {k: v.clone() for k, v in m.state_dict().items()}
    eap_safety_circuit(m, HARMFUL, HARMLESS, EAPSafetyCircuitConfig(method="eap-ig", ig_steps=2, top_k_edges=5, batch_size=3, max_seq_len=32), tokenizer=tok)
    assert n_hooks() == 0 and m.training is True
    assert all(p.grad is None for p in m.parameters()) and all(torch.equal(v, before[k]) for k, v in m.state_dict().items())


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_metric_is_read_at_the_last_real_token_for_either_padding_side(side):
    m, tok = build(padding_side=side)
    enc, last = E._tokenize_batch(tok, ["PLACEHOLDER short", "PLACEHOLDER a much longer request than the other one here"], _cfg(), "cpu")
    mask = enc["attention_mask"]
    real = [int(torch.nonzero(mask[i])[-1]) for i in range(2)]
    assert last.tolist() == real


PAIRS = ([f"PLACEHOLDER secret request number {i} please" for i in (1, 2)] + ["PLACEHOLDER secret ask", "PLACEHOLDER secret ask again now"],
         [f"PLACEHOLDER ordinary request number {i} please" for i in (1, 2)] + ["PLACEHOLDER ordinary ask", "PLACEHOLDER ordinary ask again now"])


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_padded_batch_scores_like_the_pairs_scored_one_by_one(side):
    """Pairs of different lengths in one batch (padding on either side) must give the mean of their single-pair scores."""
    m, tok = build(padding_side=side)
    h, s = PAIRS
    lens = [len(tok(x)["input_ids"]) for x in h]
    assert len(set(lens)) > 1 and lens == [len(tok(x)["input_ids"]) for x in s], lens          # padding happens, pairs are equal length
    batched = E._run_eap(m, tok, "cpu", h, s, _cfg(batch_size=len(h)))
    singles = [E._run_eap(m, tok, "cpu", [a], [b], _cfg(batch_size=1)) for a, b in zip(h, s)]
    for n in batched:
        want = sum(x[n] for x in singles) / len(singles)
        assert abs(batched[n] - want) <= 1e-4 + 1e-3 * abs(want), (side, n, batched[n], want)


def test_an_unpadded_batch_is_bit_identical_to_before():
    """Already-correct path: equal-length prompts (no padding on either side) give the same scores whichever side the tokenizer pads."""
    a = E._run_eap(*(lambda m, t: (m, t, "cpu"))(*build(padding_side="left")), HARMFUL, HARMLESS, _cfg())
    b = E._run_eap(*(lambda m, t: (m, t, "cpu"))(*build(padding_side="right")), HARMFUL, HARMLESS, _cfg())
    assert all(a[n] == b[n] for n in a)


def _reference_scores(m, tok, cfg, alphas):
    """Autograd reference: node score = sum((corrupted - clean) * mean_k dMetric/dActivation), the graph left intact, the gradient taken at the
    input embeddings corrupted + alpha * (clean - corrupted) for each alpha (EAP: alpha = 1; EAP-IG inputs: k / steps, k = 0..steps-1)."""
    nodes = E._enumerate_nodes(m, cfg.granularity)
    cid, rid = E._first_token_id(tok, cfg.compliance_token), E._first_token_id(tok, cfg.refusal_token)
    clean, cl = E._tokenize_batch(tok, HARMFUL, cfg, "cpu")
    corr, _ = E._tokenize_batch(tok, HARMLESS, cfg, "cpu")
    assert clean["input_ids"].shape == corr["input_ids"].shape
    live = {}

    def hook_all(grad=True):
        hs = []
        for nd in nodes:
            if nd.hook == "pre":
                def f(mod, args, n=nd):
                    live[n.name] = (args[0], n.channels)
                    if grad: args[0].retain_grad()
                hs.append(nd.module.register_forward_pre_hook(f))
            else:
                def f(mod, i, o, n=nd):
                    t = o[0] if isinstance(o, tuple) else o
                    live[n.name] = (t, None)
                    if grad: t.retain_grad()
                hs.append(nd.module.register_forward_hook(f))
        return hs

    def acts(enc):
        hs = hook_all(grad=False); live.clear()
        with torch.no_grad():
            m(**enc)
        [h.remove() for h in hs]
        return {k: (t if ch is None else t[..., ch[0]:ch[1]]).detach().clone() for k, (t, ch) in live.items()}

    ca, ka = acts(clean), acts(corr)
    emb = m.get_input_embeddings()
    with torch.no_grad():
        ce, ke = emb(clean["input_ids"]), emb(corr["input_ids"])
    total = {n: torch.zeros_like(ca[n]) for n in ca}
    for a in alphas:
        x = (ke + a * (ce - ke)).clone().requires_grad_(True)
        hs = hook_all(); live.clear()
        E._logit_diff_metric(m(inputs_embeds=x, attention_mask=clean["attention_mask"]).logits, cl, cid, rid).backward()
        for k, (t, ch) in live.items():
            total[k] += (t.grad if ch is None else t.grad[..., ch[0]:ch[1]]).detach()
        [h.remove() for h in hs]
        m.zero_grad()
    return {n: float(((ka[n] - ca[n]) * (total[n] / len(alphas))).sum()) / len(HARMFUL) for n in ca}


def _close(got, ref, tol=1e-4):
    scale = max(abs(v) for v in ref.values())
    worst = max(abs(got[n] - ref[n]) for n in ref)
    assert worst < tol * scale + 1e-8, f"max |code - autograd reference| = {worst:.3e} on scores of scale {scale:.3e}"


@pytest.mark.parametrize("granularity", ["block", "head"])
def test_eap_node_scores_equal_the_autograd_reference(granularity):
    m, tok = build()
    cfg = _cfg(granularity=granularity)
    _close(E._run_eap(m, tok, "cpu", HARMFUL, HARMLESS, cfg), _reference_scores(m, tok, cfg, [1.0]))


@pytest.mark.parametrize("steps", [1, 3, 5])
def test_eap_ig_node_scores_equal_the_integrated_gradient_reference(steps):
    m, tok = build()
    cfg = _cfg(method="eap-ig", ig_steps=steps)
    _close(E._run_eap(m, tok, "cpu", HARMFUL, HARMLESS, cfg), _reference_scores(m, tok, cfg, [k / steps for k in range(steps)]))


def test_the_last_block_mlp_score_is_the_direct_path_score_it_always_was():
    """The last node has no later node to pass through, so its gradient was already the true one: -0.02370 summed over 6 pairs before the per-pair mean."""
    m, tok = build()
    got = E._run_eap(m, tok, "cpu", HARMFUL, HARMLESS, _cfg())
    assert abs(got["blocks.3.mlp"] - (-0.02370 / 6)) < 2e-5


def test_earlier_nodes_differ_from_the_direct_path_scores_the_old_code_returned():
    """Regression guard: blocks.0.attn was -0.02637 (direct path only, sum of 6 pairs) and is +0.08706 with the true gradient."""
    m, tok = build()
    got = E._run_eap(m, tok, "cpu", HARMFUL, HARMLESS, _cfg())
    assert got["blocks.0.attn"] > 0 and abs(got["blocks.0.attn"] - 0.08706 / 6) < 2e-4


def test_eap_ig_with_zero_or_mean_intervention_is_refused():
    m, tok = build()
    for iv in ("zero", "mean"):
        with pytest.raises(ValueError, match="patching"):
            eap_safety_circuit(m, HARMFUL, HARMLESS, EAPSafetyCircuitConfig(method="eap-ig", intervention=iv, max_seq_len=32), tokenizer=tok)
    ci = eap_safety_circuit(m, HARMFUL, HARMLESS, EAPSafetyCircuitConfig(method="eap", intervention="zero", top_k_edges=3, batch_size=3, max_seq_len=32), tokenizer=tok)
    assert len(ci.safety_units.unit_ids) == 3


def test_a_model_whose_parameters_are_frozen_still_gets_gradients():
    m, tok = build()
    for p in m.parameters():
        p.requires_grad_(False)
    got = E._run_eap(m, tok, "cpu", HARMFUL, HARMLESS, _cfg())
    assert all(v == v for v in got.values()) and any(v != 0 for v in got.values())


def _eap_tmp_dirs():
    import glob
    import tempfile
    return set(glob.glob(f"{tempfile.gettempdir()}/safetune_eap_*"))


def test_no_temp_directory_is_left_behind_without_an_output_dir():
    m, tok = build()
    before = _eap_tmp_dirs()
    ci = eap_safety_circuit(m, HARMFUL, HARMLESS, EAPSafetyCircuitConfig(method="eap", top_k_edges=3, batch_size=3, max_seq_len=32), tokenizer=tok)
    assert _eap_tmp_dirs() == before and ci.raw_output_path is None


def test_an_output_dir_still_gets_the_scores_and_the_contrast_pairs(tmp_path):
    import json
    m, tok = build()
    ci = eap_safety_circuit(m, HARMFUL, HARMLESS, EAPSafetyCircuitConfig(method="eap", top_k_edges=3, batch_size=3, max_seq_len=32, output_dir=str(tmp_path / "o")), tokenizer=tok)
    saved = json.loads((tmp_path / "o" / "eap_scores.json").read_text())
    assert ci.raw_output_path == str(tmp_path / "o" / "eap_scores.json") and set(saved["circuit"]) == set(ci.safety_units.unit_ids)
    assert len(json.loads((tmp_path / "o" / "contrast_pairs.json").read_text())) == len(HARMFUL) and len(saved["all_scores"]) > 3


def test_the_docs_say_what_the_module_is():
    d = " ".join(E.__doc__.split())
    assert "node-level" in d and "Simplified" in d and "faithful from-scratch" not in d
    assert "sum of the scores of its outgoing edges" in d and "IG, inputs" in d and "attribute.py:149-181" in d
    c = " ".join(EAPSafetyCircuitConfig.__doc__.split())
    assert "NODES (not edges" in c and "default is 30" in c
    m, tok = build()
    ci = eap_safety_circuit(m, HARMFUL, HARMLESS, EAPSafetyCircuitConfig(method="eap", top_k_edges=3, batch_size=3, max_seq_len=32), tokenizer=tok)
    assert ci.layer_suggestions.metadata["source"] == "safetune.instrumentation.interpret.eap (native node-level EAP/EAP-IG)"
