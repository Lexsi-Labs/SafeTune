"""SafeReAct's representation-training loop and AAQ's APC calibration ask for the last position's logits only.

Both read ``hidden_states`` from every forward and never the logits, yet the LM head built a
``(batch, seq, vocab)`` tensor per forward (SafeReAct: reference, original and every LoRA step;
AAQ: the calibrated model and both references at every step). The LM head must now see one
position, and the trained weights must be exactly what the full-logits forwards gave.
"""
import torch

import safetune.core.patches.aaq_patch as aaq
import safetune.interventions.recover.safereact as sr
from tests.runtime.test_hackathon_fixes2 import _tiny_llama


def _watch(*models):
    positions, handles = [], []
    for m in models:
        handles.append(m.lm_head.register_forward_hook(lambda mod, i, o: positions.append(o.shape[1])))
    return positions, handles


def _safereact(monkeypatch, full_logits):
    if full_logits:
        monkeypatch.setattr(sr, "_last_logit_only", lambda m: {})
    model, ref = _tiny_llama(0, torch.float32), _tiny_llama(1, torch.float32)
    probes = torch.randint(1, 64, (2, 12), generator=torch.Generator().manual_seed(0))
    positions, handles = _watch(model, ref)
    ok = sr._train_reactivation_lora(model, ref, ["q_proj"], probes, lora_rank=2, lora_alpha=4.0,
                                     reactivation_scale=1.0, iters=3, lr=1e-2)
    for h in handles:
        h.remove()
    monkeypatch.undo()
    assert ok
    return positions, {n: p.detach().clone() for n, p in model.named_parameters()}


def test_safereact_reads_hidden_states_without_full_logits(monkeypatch):
    positions, got = _safereact(monkeypatch, full_logits=False)
    assert positions and set(positions) == {1}, positions
    _, ref = _safereact(monkeypatch, full_logits=True)
    assert got.keys() == ref.keys()
    assert all(torch.equal(got[n], ref[n]) for n in got)


def _aaq(monkeypatch, full_logits):
    if full_logits:
        monkeypatch.setattr(aaq, "_last_logit_only", lambda m: {})
    model = _tiny_llama(0, torch.float32)
    refs = iter([_tiny_llama(1, torch.float32).eval(), _tiny_llama(2, torch.float32).eval()])
    monkeypatch.setattr(aaq, "_try_load_hf_model", lambda path, device: next(refs))
    torch.manual_seed(0)                               # the probe ids are random
    positions, handles = _watch(model)
    patch = aaq.AAQPatch(params={"aligned_model_path": "a", "base_model_path": "b",
                                 "calibration_steps": 3, "lr": 1e-2})
    ref_heads = []
    orig = aaq._try_load_hf_model
    def load(path, device):
        m = orig(path, device)
        ref_heads.append(m.lm_head.register_forward_hook(lambda mod, i, o: positions.append(o.shape[1])))
        return m
    monkeypatch.setattr(aaq, "_try_load_hf_model", load)
    patch.apply_to_model(model)
    assert len(ref_heads) == 2                         # both references were loaded and watched
    for h in handles + ref_heads:
        h.remove()
    monkeypatch.undo()
    return positions, {n: p.detach().clone() for n, p in model.named_parameters()}


def test_aaq_reads_hidden_states_without_full_logits(monkeypatch):
    positions, got = _aaq(monkeypatch, full_logits=False)
    assert positions and set(positions) == {1}, positions
    _, ref = _aaq(monkeypatch, full_logits=True)
    assert all(torch.equal(got[n], ref[n]) for n in got)
