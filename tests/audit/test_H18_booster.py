"""Audit H18: the Booster runner used a stale harmful gradient.

Huang et al. 2025 (arXiv:2409.01586) Alg. 1 / official git-disl/Booster @8f14566
`trainer.py:272-353` (`BoosterAlignmentTrainer.training_step`): at EVERY step the harmful gradient
g_h is recomputed at the current weights, the weights are moved to w - alpha * g_h / ||g_h||, g_h' is
computed there on the same harmful batch, and the update is g_align + lambda * (g_h - g_h').
`BoosterTrainer.train` computed g_h once at the initial weights and reused it, so lambda * g_h was a
constant vector and the perturbation direction never followed the model (tiny model, 4 steps:
relative difference to the official update 0.50 / 1.11 / 1.61 from step 2 on, cosine 0.91 -> 0.58).
"""
import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import safetune
import safetune.harden as HARD
from safetune.runner.harden import BoosterTrainer

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
STEPS = 4


@pytest.fixture(scope="module")
def tok():
    t = AutoTokenizer.from_pretrained(TINY)
    t.pad_token = t.pad_token or t.eos_token
    return t


@pytest.fixture()
def no_dropout():
    old = safetune.get_config().lora_dropout
    safetune.configure(lora_dropout=0.0)          # deterministic gradients for the comparison
    yield
    safetune.configure(lora_dropout=old)


def _data(tok, n, text):
    rows = []
    for i in range(n):
        e = tok(f"{text} {i} answer text here", padding="max_length", max_length=24, return_tensors="pt")
        ids = e["input_ids"][0]
        lab = ids.clone()
        lab[:6] = -100
        lab[e["attention_mask"][0] == 0] = -100
        rows.append({"input_ids": ids.tolist(), "attention_mask": e["attention_mask"][0].tolist(),
                     "labels": lab.tolist()})
    return Dataset.from_list(rows)


def _flat(d):
    return torch.cat([v.flatten() for _, v in sorted(d.items())])


def _official_final_grad(model, align_grads, batches, alpha, lamb):
    """Independent re-implementation of git-disl/Booster trainer.py:281-353 (meta_term=True)."""
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}

    def harm_grad():
        acc = {n: torch.zeros_like(p) for n, p in params.items()}
        for b in batches:
            model.zero_grad(set_to_none=True)
            model(**b).loss.backward()
            for n, p in params.items():
                if p.grad is not None:
                    acc[n] += p.grad.detach() / len(batches)
        model.zero_grad(set_to_none=True)
        return acc

    g_h = harm_grad()
    norm = torch.sqrt(sum((g ** 2).sum() for g in g_h.values())) + 1e-7
    with torch.no_grad():
        for n, p in params.items():
            p.data -= alpha * g_h[n] / norm
    g_hp = harm_grad()
    with torch.no_grad():
        for n, p in params.items():
            p.data += alpha * g_h[n] / norm
    return {n: align_grads[n] + lamb * g_h[n] - lamb * g_hp[n] for n in align_grads}, g_h


def test_harmful_gradient_is_recomputed_at_every_step_and_matches_the_official_update(
        monkeypatch, tok, tmp_path, no_dropout):
    calls, seen_models, records = [], [], []
    orig_collect, orig_project = HARD.collect_harmful_gradient, HARD.booster_project

    def spy_collect(model, batches, *a, **k):
        calls.append(1)
        seen_models.append((model, batches))
        return orig_collect(model, batches, *a, **k)

    def spy_project(grads, harmful_grads, perturbed, cfg):
        model, batches = seen_models[-1]
        ref, ref_gh = _official_final_grad(model, grads, batches, cfg.alpha, cfg.lamb)
        out = orig_project(grads, harmful_grads, perturbed, cfg)
        records.append((out, ref, harmful_grads, ref_gh))
        return out

    monkeypatch.setattr(HARD, "collect_harmful_gradient", spy_collect)
    monkeypatch.setattr(HARD, "booster_project", spy_project)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    harm = _data(tok, 4, "harmful request")
    harmful_batches = [{k: torch.tensor([harm[i][k]]) for k in ("input_ids", "attention_mask", "labels")}
                       for i in range(4)]
    BoosterTrainer(model, tok, epochs=1, batch_size=4, lr=1e-2, results_dir=str(tmp_path)
                   ).train(_data(tok, 4 * STEPS, "user question"), out_dir=str(tmp_path / "out"),
                           harmful_batches=harmful_batches)

    assert len(calls) == 2 * STEPS, f"{len(calls)} harmful-gradient computations for {STEPS} steps"
    assert len(records) == STEPS
    for step, (out, ref, gh, ref_gh) in enumerate(records, 1):
        a, b = _flat(out), _flat(ref)
        cos = float(a @ b / (a.norm() * b.norm()))
        assert cos >= 0.999, f"step {step}: cosine {cos:.4f} to the official update"
        gcos = float(_flat(gh) @ _flat(ref_gh) / (_flat(gh).norm() * _flat(ref_gh).norm()))
        assert gcos >= 0.999, f"step {step}: g_h is stale (cosine {gcos:.4f} to g_h at the current weights)"
    # step 1: the weights are still the initial ones, so the fresh g_h is the old (initial) g_h exactly
    assert torch.equal(_flat(records[0][2]), _flat(records[0][3]))
