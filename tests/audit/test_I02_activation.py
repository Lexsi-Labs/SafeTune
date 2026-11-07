"""Audit I02 (activation mode) vs Chen et al. arXiv 2406.14144 Eq. 2 and THU-KEG/SafetyNeuron@6fc81b2 (hook_post). Random tiny Llama, placeholder prompts."""
import torch

from safetune.instrumentation.interpret import SafetyNeuronConfig, identify_safety_neurons

from _interpret_common import HARMFUL, HARMLESS, build


def _manual_mean_abs(m, tok, prompts, layer):
    """mean |input of down_proj| per neuron over all (unpadded, batch-1) tokens: the paper's MLP intermediate neurons."""
    cap = []
    h = m.model.layers[layer].mlp.down_proj.register_forward_pre_hook(lambda mod, a: cap.append(a[0].detach().float()))
    for p in prompts:
        with torch.no_grad():
            m(**tok(p, return_tensors="pt"))
    h.remove()
    t = torch.cat([c.reshape(-1, c.shape[-1]) for c in cap])
    return t.abs().mean(0)


def test_activation_mode_scores_the_input_of_down_proj():
    """The default hook used to be mlp.act_fn (sigma(W_gate x) only); the neuron of Eq. 2 is sigma(W_gate x) * W_up x, the input of down_proj."""
    m, tok = build()
    cfg = SafetyNeuronConfig(method="activation", top_k_per_layer=8, activation_batch_size=1)
    rep = identify_safety_neurons(m, {}, cfg, tokenizer=tok, harmful_prompts=HARMFUL, harmless_prompts=HARMLESS)
    assert rep.target_module == "mlp.down_proj"
    for layer in (0, 2):
        want = _manual_mean_abs(m, tok, HARMFUL, layer) - _manual_mean_abs(m, tok, HARMLESS, layer)
        top = want.abs().topk(8).indices.tolist()
        assert [i for i, _ in rep.per_layer[layer]] == top
        assert all(abs(s - float(want[i])) < 1e-5 for i, s in rep.per_layer[layer])      # signed


def test_the_old_hook_is_still_reachable_by_name_and_reads_the_output():
    m, tok = build()
    cfg = SafetyNeuronConfig(method="activation", top_k_per_layer=4, activation_module="mlp.act_fn")
    rep = identify_safety_neurons(m, {}, cfg, tokenizer=tok, harmful_prompts=HARMFUL, harmless_prompts=HARMLESS)
    assert rep.target_module == "mlp.act_fn" and sorted(rep.per_layer) == [0, 1, 2, 3]


def test_activation_mode_is_batch_size_invariant_and_leaves_no_hooks():
    m, tok = build()
    out = []
    for bs in (1, 6):
        cfg = SafetyNeuronConfig(method="activation", top_k_per_layer=3, activation_batch_size=bs)
        out.append(identify_safety_neurons(m, {}, cfg, tokenizer=tok, harmful_prompts=HARMFUL, harmless_prompts=HARMLESS))
    for l in out[0].per_layer:
        assert [i for i, _ in out[0].per_layer[l]] == [i for i, _ in out[1].per_layer[l]]
    assert sum(len(x._forward_hooks) + len(x._forward_pre_hooks) for x in m.modules()) == 0


def test_a_signed_activation_score_keeps_its_sign_and_the_ranking_is_by_magnitude():
    m, tok = build()
    for score in ("mean_diff", "mean_abs_diff", "tstat"):
        cfg = SafetyNeuronConfig(method="activation", activation_score=score, top_k_per_layer=20)
        rep = identify_safety_neurons(m, {}, cfg, tokenizer=tok, harmful_prompts=HARMFUL, harmless_prompts=HARMLESS)
        for layer, units in rep.per_layer.items():
            mags = [abs(s) for _, s in units]
            assert mags == sorted(mags, reverse=True)
        scores = [s for units in rep.per_layer.values() for _, s in units]
        assert any(s < 0 for s in scores) and any(s > 0 for s in scores), (score, "signs were dropped")


def test_the_set_of_selected_neurons_is_unchanged_by_keeping_the_sign():
    """Already-correct path: the same units are selected, in the same order, as with the old |score| output."""
    m, tok = build()
    cfg = SafetyNeuronConfig(method="activation", top_k_per_layer=5)
    rep = identify_safety_neurons(m, {}, cfg, tokenizer=tok, harmful_prompts=HARMFUL, harmless_prompts=HARMLESS)
    cfg2 = SafetyNeuronConfig(method="activation", top_k_per_layer=5, abs_rank=False)       # always returned the signed score
    rep2 = identify_safety_neurons(m, {}, cfg2, tokenizer=tok, harmful_prompts=HARMFUL, harmless_prompts=HARMLESS)
    assert rep.per_layer == rep2.per_layer
    cfg3 = SafetyNeuronConfig(method="activation", top_k_per_layer=5, activation_score="mean_abs_diff")
    assert [[i for i, _ in u] for u in rep.per_layer.values()] == [[i for i, _ in u] for u in identify_safety_neurons(
        m, {}, cfg3, tokenizer=tok, harmful_prompts=HARMFUL, harmless_prompts=HARMLESS).per_layer.values()]
