"""The SafeGradTrainer runner passes gradient accumulation and surgery_per_optimizer_step through (#80)."""
import pytest
import torch
from datasets import Dataset

import safetune.config as sc
import safetune.harden as HARD
from safetune.config import configure
from safetune.runner.harden import SafeGradTrainer

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch):
    monkeypatch.setattr(sc, "_RUNTIME", sc.RuntimeConfig())
    configure(device="cpu", dtype="float32")


def _rows(n, seed):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(n):
        ids = torch.randint(5, 1000, (12,), generator=g).tolist()
        rows.append({"input_ids": ids, "attention_mask": [1] * 12, "labels": ids})
    return Dataset.from_list(rows)


def test_runner_accumulates_and_runs_surgery_once_per_step(monkeypatch, tmp_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.pad_token or tok.eos_token
    seen = []

    class Spy(HARD.SafeGradHFTrainer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.surgeries = 0
            seen.append(self)

        def _apply_global_surgery(self, model, g_align):
            self.surgeries += 1
            return super()._apply_global_surgery(model, g_align)

    monkeypatch.setattr(HARD, "SafeGradHFTrainer", Spy)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    SafeGradTrainer(model, tok, batch_size=1, results_dir=str(tmp_path),
                    gradient_accumulation_steps=2, surgery_per_optimizer_step=True,
                    ).train(_rows(4, 0), out_dir=str(tmp_path / "out"), safety_dataset=_rows(2, 1))
    tr = seen[0]
    assert tr.args.gradient_accumulation_steps == 2
    assert tr.state.global_step == 2                 # 4 rows, batch 1, accumulation 2
    assert tr.surgeries == 2
