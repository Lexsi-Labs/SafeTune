"""Lightweight end-to-end proxy smoke for all four intervention pillars.

Runs only in proxy mode (``SAFETUNE_PROXY=1``); see conftest.py. Each test takes
a small public model through a real train/apply/calibrate/unlearn step and then
generates from the resulting model, proving the whole pipeline executes without
the big drifted checkpoints. These are NOT drift-metric tests — they assert the
pipeline runs and produces a usable model, not that safety improved.
"""
import os

import pytest
import torch

from safetune.runner.utils.model_utils import load_tok, load_model, load_model_cpu

pytestmark = pytest.mark.proxy

_DRIFT = os.environ.get("SAFETUNE_DRIFT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
_BASE = os.environ.get("SAFETUNE_BASE_MODEL", "Qwen/Qwen2-0.5B")
_ALIGNED = os.environ.get("SAFETUNE_ALIGNED_MODEL", "Qwen/Qwen2-0.5B-Instruct")


def _to_gpu(m):
    return m.to("cuda") if torch.cuda.is_available() else m


def _slice(ds, n=8):
    try:
        return ds.select(range(min(n, len(ds))))
    except Exception:
        return ds


def _assert_generates(model, tok, msg):
    """Cheap eval: the pipeline's output model must generate NEW tokens.

    Decodes only the continuation (excluding the echoed prompt) so the assertion
    can actually fail — a model that emits nothing but the prompt is a failure.
    """
    model.eval()
    dev = next(model.parameters()).device
    inp = tok("Explain safety in one sentence:", return_tensors="pt").to(dev)
    n_prompt = inp.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    assert out.shape[1] > n_prompt, f"{msg}: no new tokens generated"
    cont = tok.decode(out[0][n_prompt:], skip_special_tokens=True)
    assert cont.strip(), f"{msg}: continuation decoded to empty text"


def test_proxy_harden():
    from safetune.runner.harden import SafeGradTrainer, load_harden_data
    tok = load_tok(_DRIFT)
    train_set, safety_set = load_harden_data(_DRIFT, n=8)
    tr = SafeGradTrainer(model_id=_DRIFT, model=_to_gpu(load_model(_DRIFT)),
                         tokenizer=tok, epochs=1, batch_size=2)
    out = tr.train(_slice(train_set), out_dir="proxy_harden",
                   safety_dataset=_slice(safety_set))
    assert out and os.path.isdir(out), "harden did not produce a checkpoint dir"
    # train -> save -> reload -> infer
    _assert_generates(_to_gpu(load_model(out)), tok, "harden")


def test_proxy_recover():
    from safetune.runner.recover import ReStaTrainer
    tok = load_tok(_DRIFT)
    tr = ReStaTrainer(model_id=_DRIFT, model=_to_gpu(load_model(_DRIFT)),
                      base_model=load_model_cpu(_BASE),
                      aligned_model=load_model_cpu(_ALIGNED),
                      alpha=0.5, dare=True, dare_seed=0)
    patched = tr.apply()
    assert patched is not None, "recover.apply() returned None"
    _assert_generates(patched, tok, "recover")


def test_proxy_unlearn():
    from safetune.runner.unlearn import NPOTrainer, load_unlearn_data
    tok = load_tok(_DRIFT)
    forget, retain = load_unlearn_data(_DRIFT)
    tr = NPOTrainer(model_id=_DRIFT, model=_to_gpu(load_model(_DRIFT)),
                    variant="npo_grad_diff", beta=0.1, num_epochs=1, lr=1e-5)
    unlearned = tr.unlearn(_slice(forget), _slice(retain))
    assert unlearned is not None, "unlearn() returned None"
    _assert_generates(unlearned, tok, "unlearn")


def test_proxy_steer():
    from safetune.runner.steer import CAATrainer, load_steer_data
    tok = load_tok(_DRIFT)
    harmful, harmless = load_steer_data(16)
    tr = CAATrainer(model_id=_DRIFT, model=_to_gpu(load_model(_DRIFT)),
                    tokenizer=tok, multiplier=5.0)
    caa_model, _ = tr.calibrate(harmful, harmless)
    assert caa_model is not None, "steer.calibrate() returned None"
    _assert_generates(caa_model, tok, "steer")
