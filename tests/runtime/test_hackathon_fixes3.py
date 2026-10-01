"""Hackathon fixes, part 3 (validation report). CPU only, no downloads: tokenizers
are built in memory, models are tiny random configs."""
import os
import subprocess
import sys

import pytest

from .test_hackathon_cohere_fixes import _word_tok
from .test_hackathon_fixes2 import _tiny_llama


# ── 2. eap_safety_circuit on an already-loaded model ─────────────────────────

def test_eap_safety_circuit_uses_a_loaded_model_in_place(monkeypatch):
    import torch
    import safetune._refusal_helpers as rh
    from safetune.core.circuit_kit.interface import CircuitInfo
    from safetune.interpret import EAPSafetyCircuitConfig, eap_safety_circuit

    def no_second_copy(*a, **k):
        raise AssertionError("eap_safety_circuit loaded a second copy of the model")
    monkeypatch.setattr(rh, "_load_pretrained_lm", no_second_copy)

    tok = _word_tok()
    model = _tiny_llama(0, torch.float32)
    model.resize_token_embeddings(len(tok))
    model.train()
    cfg = EAPSafetyCircuitConfig(method="eap-ig", ig_steps=2, top_k_edges=4, batch_size=2,
                                 max_seq_len=16, compliance_token="paris", refusal_token="bread")
    info = eap_safety_circuit(model, ["how do i bake bread"] * 2, ["what is the capital of france"] * 2,
                              cfg, tokenizer=tok)
    assert isinstance(info, CircuitInfo) and len(info.safety_units.unit_ids) == 4
    assert model.training  # caller's mode restored
    assert all(p.grad is None for p in model.parameters())


# ── 3. configurable eval split ───────────────────────────────────────────────

@pytest.mark.parametrize("split, config, expected", [
    (None, None, "validation"), (None, "main", "test"), ("train", None, "train")])
def test_eval_task_split_is_configurable(monkeypatch, split, config, expected):
    from safetune.core.eval.core import EvalRegistry, EvalRunner, EvalTask, TaskCategory
    seen = {}
    monkeypatch.setitem(EvalRegistry._datasets, "toy", lambda config, split: seen.setdefault("split", split))
    kw = {} if split is None else {"split": split}
    task = EvalTask(name="t", category=list(TaskCategory)[0], description="", dataset_name="toy",
                    dataset_config=config, **kw)
    EvalRunner._load_dataset(None, task)
    assert seen["split"] == expected


# ── 3b. configurable ASRT prompt cap ─────────────────────────────────────────

def test_asrt_default_prompts_capped_by_argument(monkeypatch):
    import safetune.runner.utils.data_utils as du
    from safetune.harden.asrt import ASRTCallback
    monkeypatch.setattr(du, "load_bench_prompts", lambda b: {"harmbench": [f"p{i}" for i in range(100)]})
    assert len(ASRTCallback(attacker=object()).adversarial_prompts) == 64
    cb = ASRTCallback(attacker=object(), n_adversarial_prompts=5)
    assert list(cb.adversarial_prompts) == ["p0", "p1", "p2", "p3", "p4"]
    assert len(ASRTCallback(object(), ["a", "b"]).adversarial_prompts) == 2  # explicit list unchanged


# ── 4. packaging: `safetune list | head` ─────────────────────────────────────

def test_cli_list_into_closed_pipe_has_no_traceback():
    r, w = os.pipe()
    os.close(r)  # the reader (`head`) has already gone
    p = subprocess.run([sys.executable, "-c", "from safetune.cli import main; main()", "list"],
                       stdout=w, stderr=subprocess.PIPE, text=True, timeout=300)
    os.close(w)
    assert "Traceback" not in p.stderr and "BrokenPipeError" not in p.stderr, p.stderr
    assert p.returncode == 1


# ── 5. evaluate() reports the benchmark size next to n_evaluated ─────────────

def test_evaluate_reports_n_total_and_n_evaluated(monkeypatch):
    import importlib
    ev = importlib.import_module("safetune.instrumentation.evaluate.suite.evaluate")
    import safetune.instrumentation.evaluate.suite.benchmarks as bm
    monkeypatch.setattr(bm, "load_benchmark", lambda name: [{"prompt": f"q{i}"} for i in range(10)])
    monkeypatch.setattr(ev, "_compute_metrics",
                        lambda *, max_prompts, **kw: {"asr": 0.0, "n_evaluated": max_prompts})
    out = ev.evaluate(object(), benchmarks=["harmbench"], max_prompts=4)["harmbench"]
    assert out["n_total"] == 10 and out["n_evaluated"] == 4
    assert out["n"] == out["n_total"]  # deprecated alias kept


# ── 1. AlphaSteer in fp16 ────────────────────────────────────────────────────

def test_alphasteer_fp16_steering_stays_finite():
    import torch
    from safetune.steer import AlphaSteerModel
    model = _tiny_llama(0, torch.float16)
    torch.manual_seed(1)
    harmful, benign = torch.randn(16, 64, dtype=torch.float16) + 3, torch.randn(16, 64, dtype=torch.float16)
    ids = torch.randint(0, 64, (2, 6))
    # strength large enough that the steered state overflows fp16 (as the
    # default alpha=20 over 9 layers did on North Micro Vision)
    with AlphaSteerModel(model, harmful, benign, layer_id=2, strength=1e8) as steered:
        with torch.no_grad():
            logits = steered(input_ids=ids).logits
    assert torch.isfinite(logits).all()
