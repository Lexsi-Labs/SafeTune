"""Audit H27: the LoXHarden runner extrapolates the BASE, and a missing reference turns it into plain SFT.

Perin et al. 2025 (arXiv:2506.15606) and the official VITA-Group/LoX @d2411de `safety/LoX.py`:
    new = W_aligned + coef * lowrank_k(W_aligned - W_base)          (k=6, coef=1.25 in the experiments)
`apply_lox_harden(model, base_sd, aligned_sd)` computes exactly that when `model` is the aligned model.
`LoXHardenTrainer.train` used to pass `self.model.state_dict()` as the BASE and load the aligned model from
`aligned_model_path`, so it applied `W_base + alpha * lowrank(W_aligned - W_base)` to the model it then
fine-tunes (max |W - LoX(aligned)| = 0.0951 on the tiny model). Without `aligned_model_path` the reference was
the training model itself: delta = 0, a no-op, with only a warning. Now (H10 convention): the trained model is
the aligned one, `base_model_path` is required, `aligned_model_path` raises.
"""
import copy

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import safetune.harden as H
from safetune.runner.harden import LoXHardenTrainer
from safetune.runner.harden import _other as other

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _official(aligned_sd, base_sd, k, coef):
    out = {}
    for name, wa in aligned_sd.items():
        if wa.dim() > 1:
            d = wa.float() - base_sd[name].float()
            U, S, Vt = torch.linalg.svd(d, full_matrices=False)
            S[k:] = 0
            out[name] = wa.float() + coef * (U @ torch.diag(S) @ Vt)
    return out


def _models():
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    aligned = copy.deepcopy(base)
    with torch.no_grad():
        for p in aligned.parameters():
            p.add_(0.02 * torch.randn_like(p))
    return base, aligned


def test_apply_lox_harden_on_the_aligned_model_equals_the_official_script():
    base, aligned = _models()
    want = _official(aligned.state_dict(), base.state_dict(), 6, 1.25)
    model = copy.deepcopy(aligned)
    H.apply_lox_harden(model, base.state_dict(), aligned.state_dict(), H.LoXHardenConfig(alpha=1.25, rank=6))
    for n, p in model.named_parameters():
        if n in want:
            assert torch.allclose(p, want[n], atol=1e-5), n


class _Stop(Exception):
    pass


def _dataset():
    from datasets import Dataset

    ids = [[5, 6, 7, 8, 9, 10]] * 4
    return Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * 6] * 4, "labels": ids})


def _run_runner(monkeypatch, tmp_path, trained, **kwargs):
    seen = {}

    class StubTrainer:
        def __init__(self, model=None, **kw):
            seen["model"] = model

        def train(self):
            raise _Stop

    monkeypatch.setattr(other, "Trainer", StubTrainer)
    tok = AutoTokenizer.from_pretrained(TINY)
    tr = LoXHardenTrainer(trained, tok, rank=6, extrapolation_factor=1.25, results_dir=str(tmp_path), **kwargs)
    with pytest.raises(_Stop):
        tr.train(_dataset(), out_dir=str(tmp_path / "out"))
    return seen["model"]


def test_runner_result_is_the_aligned_model_extrapolated(monkeypatch, tmp_path):
    base, aligned = _models()
    base.save_pretrained(tmp_path / "base")
    want = _official(aligned.state_dict(), base.state_dict(), 6, 1.25)
    got = _run_runner(monkeypatch, tmp_path, copy.deepcopy(aligned), base_model_path=str(tmp_path / "base"))
    worst = max((p - want[n]).abs().max().item() for n, p in got.named_parameters() if n in want)
    assert worst < 1e-5, f"max |W - LoX(aligned)| = {worst:.6f}"


def test_missing_base_model_path_raises_instead_of_training_without_lox(monkeypatch, tmp_path):
    _, aligned = _models()
    with pytest.raises(ValueError, match="base_model_path"):
        _run_runner(monkeypatch, tmp_path, copy.deepcopy(aligned))


def test_aligned_model_path_is_rejected_and_points_to_base_model_path(tmp_path):
    _, aligned = _models()
    tok = AutoTokenizer.from_pretrained(TINY)
    with pytest.raises(ValueError, match="base_model_path"):
        LoXHardenTrainer(aligned, tok, aligned_model_path=str(tmp_path / "aligned"), results_dir=str(tmp_path))


def _cli_train(monkeypatch, tmp_path, algo, *flags):
    import json
    import sys

    from safetune import cli

    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps({"prompt": f"question {i}?", "response": f"answer {i} is here"}) + "\n"
                            for i in range(4)))
    monkeypatch.setattr(sys, "argv", ["safetune", "train", "--algo", algo, "--model", TINY, "--train-dataset",
                                      str(data), "--batch-size", "2", "--output", str(tmp_path / "out"), *flags])
    cli.main()


def test_cli_base_flag_is_the_base_model_path(monkeypatch, tmp_path):
    _cli_train(monkeypatch, tmp_path, "loxharden", "--base", TINY)
    assert (tmp_path / "out" / "config.json").exists()


def test_cli_forwards_base_only_to_trainers_that_take_base_model_path(monkeypatch, tmp_path):
    """Generic on the signature, so AsFT (which also takes base_model_path) gets it too."""
    from safetune import cli
    from safetune.runner import harden

    seen = {}

    def fake(name, takes_base):
        def __init__(self, model, tok, *, base_model_path=None, **kwargs):
            seen[name] = kwargs if not takes_base else {**kwargs, "base_model_path": base_model_path}

        def __init_plain__(self, model, tok, **kwargs):
            seen[name] = kwargs

        def train(self, train_dataset, out_dir=None):
            return out_dir

        return type(name, (), {"__init__": __init__ if takes_base else __init_plain__, "train": train,
                               "dataset_inputs": []})

    for name, takes in (("TakesBase", True), ("NoBase", False)):
        monkeypatch.setattr(harden, name, fake(name, takes), raising=False)
        monkeypatch.setitem(cli.HARDEN_REGISTRY, name.lower(), name)
        _cli_train(monkeypatch, tmp_path, name.lower(), "--base", "some/base")
    assert seen["TakesBase"]["base_model_path"] == "some/base"
    assert "base_model_path" not in seen["NoBase"]
