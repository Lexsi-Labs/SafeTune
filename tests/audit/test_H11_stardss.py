"""Audit H11: STAR-DSS runner turns a missing `kind` column into V_safe = 0 on every row.

Eq. (3): L = sum V_safe * CE + (1 - V_safe) * lambda_KL * KL(pi || pi_ref). With V_safe = 0 on all
rows the loss is KL to a reference that equals the model at step 0: loss 0.0, grad_norm 0.0 at every
step and the weights never move. The runner sets V_safe = 1.0 only for rows whose `kind` is "benign",
so a dataset without that column trains nothing, silently.
"""
import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from safetune.runner.harden import STARDSSTrainer

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _data(tok, with_kind, kind=None):
    rows = []
    for i in range(8):
        e = tok(f"user question {i} answer text here", padding="max_length", max_length=16,
                return_tensors="pt")
        ids = e["input_ids"][0]
        lab = ids.clone()
        lab[:6] = -100
        lab[e["attention_mask"][0] == 0] = -100
        r = {"input_ids": ids.tolist(), "attention_mask": e["attention_mask"][0].tolist(),
             "labels": lab.tolist()}
        if with_kind:
            r["kind"] = kind or ("benign" if i % 2 == 0 else "harmful")
        rows.append(r)
    return Dataset.from_list(rows)


@pytest.fixture(scope="module")
def tok():
    t = AutoTokenizer.from_pretrained(TINY)
    t.pad_token = t.pad_token or t.eos_token
    return t


def _weights_seen_by_trainer(monkeypatch, tok, tmp_path, with_kind, kind=None):
    """Run the runner with a spy trainer; return the safety_weights column it receives."""
    import safetune.harden as HARD

    seen = {}
    orig = HARD.STARDSSHFTrainer

    class Spy(orig):
        def __init__(self, *a, **k):
            seen["ds"] = k["train_dataset"]
            super().__init__(*a, **k)

    monkeypatch.setattr(HARD, "STARDSSHFTrainer", Spy)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    STARDSSTrainer(model, tok, epochs=1, batch_size=2, lr=1e-2, results_dir=str(tmp_path)
                   ).train(_data(tok, with_kind, kind), out_dir=str(tmp_path / "out"))
    return torch.stack([torch.as_tensor(r["safety_weights"]) for r in seen["ds"]])


def test_dataset_without_kind_is_rejected_not_trained_with_zero_weights(monkeypatch, tok, tmp_path):
    """V_safe = 0 on every row makes the loss pure KL to a reference equal to the model: loss 0.0,
    grad 0.0, weights never move (CPU, 4 steps). The runner must raise instead."""
    import pytest

    with pytest.raises(ValueError, match="kind"):
        _weights_seen_by_trainer(monkeypatch, tok, tmp_path, with_kind=False)


def test_dataset_with_kind_gets_mixed_weights(monkeypatch, tok, tmp_path):
    w = _weights_seen_by_trainer(monkeypatch, tok, tmp_path, with_kind=True)
    assert set(w.unique().tolist()) == {0.0, 1.0}


@pytest.mark.parametrize("kind,match", [("harmful", "benign"), ("safe", "safe")])
def test_kind_column_without_benign_rows_or_with_other_values_is_rejected(monkeypatch, tok, tmp_path, kind, match):
    """All rows harmful, or a value other than benign/harmful on every row, is V_safe = 0 everywhere."""
    with pytest.raises(ValueError, match=match):
        _weights_seen_by_trainer(monkeypatch, tok, tmp_path, with_kind=True, kind=kind)
