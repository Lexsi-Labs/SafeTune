"""Audit H10: the AsFT runner built no projection matrices, so the regulariser never ran.

The runner took the base weights from the LoRA-wrapped model (keys `...q_proj.base_layer.weight`)
while the trainer looked up `...q_proj.weight`, so no matrix matched and the run was plain LoRA SFT.
The runner also used `aligned_path - trained model` as the direction, which matches neither the
authors' script (it trains the aligned chat model with d = W_chat - W_base,
PKU-YuanGroup/AsFT @a811f00, AsFT_finetuning.py:84, scripts/*/AsFT_reg1_*.sh:12) nor anything useful.
Now the trained model is the aligned model and `base_model_path` is the base.
"""
import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import safetune.harden as HARD
from safetune.runner.harden import AsFTTrainer

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
_ASFT = HARD.AsFTHFTrainer


@pytest.fixture(scope="module")
def tok():
    t = AutoTokenizer.from_pretrained(TINY)
    t.pad_token = t.pad_token or t.eos_token
    return t


@pytest.fixture()
def checkpoints(tmp_path, tok):
    """base on disk, plus the aligned model (= base + a fixed shift of q/v) in memory."""
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    base.save_pretrained(tmp_path / "base")
    aligned = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in aligned.named_parameters():
            if n.endswith(("q_proj.weight", "v_proj.weight")):
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    return tmp_path, base, aligned


def _data(tok):
    ids = tok("hello world, how are you today?", padding="max_length", max_length=16,
              return_tensors="pt")
    row = {"input_ids": ids["input_ids"][0], "attention_mask": ids["attention_mask"][0],
           "labels": ids["input_ids"][0].clone()}
    return Dataset.from_list([row] * 8)


def _run(monkeypatch, tmp_path, tok, model, **kw):
    """Run the runner for 4 optimizer steps; return the trainer and [(penalty, grad_norm)] per step."""
    seen, log = [], []

    class Spy(_ASFT):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen.append(self)

        def _asft_reg_loss(self, m):
            reg = super()._asft_reg_loss(m)
            self._last_pen = 0.0 if reg is None else float(reg)
            return reg

        def training_step(self, m, inputs, num_items_in_batch=None):
            self._last_pen = 0.0
            out = super().training_step(m, inputs, num_items_in_batch)
            g = torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters() if p.grad is not None))
            log.append((self._last_pen, float(g)))
            return out

    monkeypatch.setattr(HARD, "AsFTHFTrainer", Spy)
    monkeypatch.setattr(AsFTTrainer, "HF_TRAINER", Spy)
    torch.manual_seed(0)                                   # same LoRA init in every run
    AsFTTrainer(model, tok, epochs=2, batch_size=4, lr=1e-2, results_dir=str(tmp_path), **kw
                ).train(_data(tok), out_dir=str(tmp_path / "out"))
    return seen[0], log


def test_direction_is_aligned_minus_base_and_regulariser_is_active(monkeypatch, checkpoints, tok):
    tmp_path, base, aligned = checkpoints
    expected = {n: (p.detach() - dict(base.named_parameters())[n].detach())
                for n, p in aligned.named_parameters() if n.endswith(("q_proj.weight", "v_proj.weight"))}
    tr, _ = _run(monkeypatch, tmp_path, tok, aligned, base_model_path=str(tmp_path / "base"))
    assert tr._use_lora_path and len(tr._proj_matrices) > 0
    # d = W_aligned - W_base (not the reverse): compare the dicts the runner handed over
    for key, d in expected.items():
        a = tr._aligned_sd.get(key)
        b = tr._base_sd.get(key)
        assert a is not None and b is not None, key
        torch.testing.assert_close((a.float() - b.float()), d.float())
        assert (a.float() - b.float()).abs().max() > 0
    # (I - C) for one matrix equals the closed form
    name, proj = next(iter(tr._proj_matrices.items()))
    key = name.split("base_model.model.")[-1].split(".lora_A")[0] + ".weight"
    d = expected[key].float()
    torch.testing.assert_close(proj, torch.eye(d.shape[0]) - d @ d.t() / d.norm())


def test_penalty_and_grad_norm_per_step_with_and_without_the_regulariser(monkeypatch, checkpoints, tok):
    tmp_path, base, aligned = checkpoints
    import copy
    _, with_reg = _run(monkeypatch, tmp_path, tok, copy.deepcopy(aligned),
                       base_model_path=str(tmp_path / "base"))
    _, without = _run(monkeypatch, tmp_path, tok, copy.deepcopy(aligned))
    print("with regulariser   (penalty, grad_norm):", [(round(p, 6), round(g, 4)) for p, g in with_reg])
    print("without regulariser(penalty, grad_norm):", [(round(p, 6), round(g, 4)) for p, g in without])
    assert len(with_reg) >= 3 and len(without) >= 3
    assert with_reg[0][0] == 0.0                     # lora_B starts at zero: no penalty at step 1
    assert all(p > 0 for p, _ in with_reg[1:3])      # B has moved: the penalty is live
    assert all(p == 0.0 for p, _ in without)
    assert any(abs(a[1] - b[1]) > 1e-6 for a, b in zip(with_reg[1:3], without[1:3]))


def test_zero_direction_raises(monkeypatch, checkpoints, tok):
    tmp_path, base, aligned = checkpoints
    aligned.load_state_dict(base.state_dict())       # W_aligned == W_base
    with pytest.raises(ValueError, match="no usable alignment direction"):
        _run(monkeypatch, tmp_path, tok, aligned, base_model_path=str(tmp_path / "base"))


def test_aligned_model_path_is_rejected(checkpoints, tok):
    tmp_path, base, aligned = checkpoints
    with pytest.raises(ValueError, match="base_model_path"):
        AsFTTrainer(aligned, tok, aligned_model_path=str(tmp_path / "base"))
