#!/usr/bin/env python3
"""Steer pillar — full end-to-end test for every trainer.

Each test runs .calibrate() followed by .evaluate() on llama31-8b-gsm8k-drift-merged
and reports PASS / FAIL / SKIP.

Usage::

    python examples/test_steer.py
"""

from __future__ import annotations
import os
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

# ── Model paths ───────────────────────────────────────────────────────────────

# ── External-checkpoint requirement ──
# Provide checkpoints via env vars:
#
#   SAFETUNE_DRIFT_MODEL   — drifted model (required)
#   SAFETUNE_BASE_MODEL    — pre-alignment base (recover)
#   SAFETUNE_ALIGNED_MODEL — aligned reference (recover)
#
#   SAFETUNE_DRIFT_MODEL — path/HF-id of the drifted (gsm8k) model to steer
#
# The harmful/harmless calibration data IS public and downloads automatically.

# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# _MERGED     = os.path.abspath(os.path.join(_SCRIPT_DIR, "../audit/training/merged"))
# _MODEL_ID   = os.environ.get("SAFETUNE_DRIFT_MODEL",
#                              os.path.join(_MERGED, "llama31-8b-gsm8k-drift-merged"))

_MODEL_ID   = "Lexsi/llama32-3b-gsm8k-sft-drift" # task aligned model: gsm8k, code, legal, medical 
_ALIGNED_ID = "Lexsi/llama32-3b-hh-rlhf-aligned" # dpo aligned model
_DOMAIN = "gsm8k" #domain must match fine tuned model task

# ── PASS / FAIL / SKIP framework ─────────────────────────────────────────────

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def _check(results: list, name: str, fn):
    try:
        fn()
        results.append((name, PASS))
        print(f"  [{PASS}] {name}")
    except ImportError as e:
        results.append((name, SKIP))
        print(f"  [{SKIP}] {name}: {e}")
    except Exception as e:
        results.append((name, FAIL))
        print(f"  [{FAIL}] {name}: {e}")
        traceback.print_exc()


# ── Trainer test functions ────────────────────────────────────────────────────

def _test_refusal_direction(fresh, tok, harmful, harmless):
    from safetune.runner.steer import RefusalDirectionTrainer
    t = RefusalDirectionTrainer(model_id=_MODEL_ID, model=fresh(),
                                tokenizer=tok, alpha=20.0, orthogonalize=False)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_caa(fresh, tok, harmful, harmless):
    from safetune.runner.steer import CAATrainer
    t = CAATrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok,
                   target_layers=list(range(14, 19)), pool_method="mean",
                   normalize=True, multiplier=20.0)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_linear_probe_guard(fresh, tok, harmful, harmless):
    from safetune.runner.steer import LinearProbeGuardTrainer
    t = LinearProbeGuardTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok,
                                layer=15, threshold=0.5)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_scans(fresh, tok, harmful, harmless):
    from safetune.runner.steer import SCANSTrainer
    t = SCANSTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_sta(fresh, tok, harmful, harmless):
    from safetune.runner.steer import STATrainer
    t = STATrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_circuit_breaker_rr(fresh, tok, harmful, harmless):
    from safetune.runner.steer import CircuitBreakerRRTrainer
    t = CircuitBreakerRRTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok,
                                threshold=0.5)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_cast(fresh, tok, harmful, harmless):
    from safetune.runner.steer import CASTTrainer
    t = CASTTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok,
                    probe_layer=15, alpha=10.0)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_adasteer(fresh, tok, harmful, harmless):
    from safetune.runner.steer import AdaSteerTrainer
    t = AdaSteerTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok,
                        alpha=15.0)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_safe_switch(fresh, tok, harmful, harmless):
    from safetune.runner.steer import SafeSwitchTrainer
    t = SafeSwitchTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok,
                          gate_layer=16, threshold=0.5)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_safe_decoding(fresh, tok, harmful, harmless):
    from safetune.runner.steer import SafeDecodingTrainer
    t = SafeDecodingTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok,
                            alpha=2.0, window=5)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_contrastive_decoding(fresh, tok, harmful, harmless):
    from safetune.runner.steer import ContrastiveDecodingTrainer
    t = ContrastiveDecodingTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok,
                                   alpha=1.0, temperature=1.0)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_proxy_tuning(fresh, tok, harmful, harmless):
    from safetune.runner.steer import ProxyTuningTrainer
    # Use the same model as proxy (degenerate but exercises the pipeline).
    proxy = fresh()
    t = ProxyTuningTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok,
                           proxy_model=proxy, alpha=1.0)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_nudging(fresh, tok, harmful, harmless):
    from safetune.runner.steer import NudgingTrainer
    t = NudgingTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok,
                       nudge_strength=5.0)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_circuit_breaker(fresh, tok, harmful, harmless):
    from safetune.runner.steer import CircuitBreakerTrainer
    t = CircuitBreakerTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_repbend(fresh, tok, harmful, harmless):
    from safetune.runner.steer import RepBendTrainer
    t = RepBendTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_tar_steer(fresh, tok, harmful, harmless):
    from safetune.runner.steer import TARSteerTrainer
    t = TARSteerTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_rrfa_ensemble(fresh, tok, harmful, harmless):
    from safetune.runner.steer import RRFAEnsembleTrainer
    t = RRFAEnsembleTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_safe_steer(fresh, tok, harmful, harmless):
    from safetune.runner.steer import SafeSteerTrainer
    t = SafeSteerTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok,
                         alpha=15.0)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")

def _test_alpha_steer(fresh, tok, harmful, harmless):
    from safetune.runner.steer import AlphaSteerTrainer
    t = AlphaSteerTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok,
                          alpha=20.0)
    wrapped, _ = t.calibrate(harmful, harmless)
    t.evaluate(wrapped, domain="gsm8k", backend="vllm")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from safetune.runner.steer import load_steer_data
    from safetune.runner.utils.model_utils import load_tok, load_model

    print("=" * 60)
    print("safetune · steer full test  (llama31-8b-gsm8k-drift-merged)")
    print("=" * 60)
    print("\nLoading calibration data and tokenizer …")
    harmful, harmless = load_steer_data(256)
    tok = load_tok(_MODEL_ID)
    print("Calibration data ready.\n")

    def fresh():
        return load_model(_MODEL_ID)

    results = []
    tests = [
        ("RefusalDirectionTrainer",   _test_refusal_direction),
        ("CAATrainer",                _test_caa),
        ("LinearProbeGuardTrainer",   _test_linear_probe_guard),
        ("SCANSTrainer",              _test_scans),
        ("STATrainer",                _test_sta),
        ("CircuitBreakerRRTrainer",   _test_circuit_breaker_rr),
        ("CASTTrainer",               _test_cast),
        ("AdaSteerTrainer",           _test_adasteer),
        ("SafeSwitchTrainer",         _test_safe_switch),
        ("SafeDecodingTrainer",       _test_safe_decoding),
        ("ContrastiveDecodingTrainer",_test_contrastive_decoding),
        ("ProxyTuningTrainer",        _test_proxy_tuning),
        ("NudgingTrainer",            _test_nudging),
        ("CircuitBreakerTrainer",     _test_circuit_breaker),
        ("RepBendTrainer",            _test_repbend),
        ("TARSteerTrainer",           _test_tar_steer),
        ("RRFAEnsembleTrainer",       _test_rrfa_ensemble),
        ("SafeSteerTrainer",          _test_safe_steer),
        ("AlphaSteerTrainer",         _test_alpha_steer),
    ]
    for name, fn in tests:
        _check(results, name,
               lambda fn=fn: fn(fresh, tok, harmful, harmless))

    total   = len(results)
    passed  = sum(1 for _, s in results if s == PASS)
    skipped = sum(1 for _, s in results if s == SKIP)
    failed  = sum(1 for _, s in results if s == FAIL)

    print(f"\n{'='*60}")
    print(f"Steer  {passed} PASS  {skipped} SKIP  {failed} FAIL  ({total} total)")
    print("=" * 60)

    if failed:
        print("\nFailed trainers:")
        for name, status in results:
            if status == FAIL:
                print(f"  {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
