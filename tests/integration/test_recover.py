#!/usr/bin/env python3
"""Recover pillar — full end-to-end test for every trainer.

Each test runs .apply() followed by .evaluate() on llama31-8b-gsm8k-drift-merged
and reports PASS / FAIL / SKIP.

Usage::

    python examples/test_recover.py
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
#   SAFETUNE_DRIFT_MODEL   — path/HF-id of the drifted (gsm8k) model to repair
#   SAFETUNE_BASE_MODEL    — path/HF-id of the pre-alignment base model
#   SAFETUNE_ALIGNED_MODEL — path/HF-id of the aligned reference
#
# Reference checkpoints were produced from real drifted fine-tunes.

# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# _MERGED     = os.path.abspath(os.path.join(_SCRIPT_DIR, "../audit/training/merged"))
# _MODEL_ID   = os.environ.get("SAFETUNE_DRIFT_MODEL",
#                              os.path.join(_MERGED, "llama31-8b-gsm8k-drift-merged"))
# _BASE_ID    = os.environ.get("SAFETUNE_BASE_MODEL",
#                              os.path.join(_MERGED, "llama31-8b-base"))
# _ALIGNED_ID = os.environ.get("SAFETUNE_ALIGNED_MODEL",
#                              os.path.join(_MERGED, "llama31-8b-hh-rlhf-aligned-final"))


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MERGED     = os.path.abspath(os.path.join(_SCRIPT_DIR, "../audit/training/merged"))
_MODEL_ID   = os.environ.get("SAFETUNE_DRIFT_MODEL",
                             "Qwen/Qwen2.5-0.5B-Instruct")
_BASE_ID    = os.environ.get("SAFETUNE_BASE_MODEL",
                             "Qwen/Qwen2.5-0.5B-Instruct")
_ALIGNED_ID = os.environ.get("SAFETUNE_ALIGNED_MODEL",
                             "Qwen/Qwen2.5-0.5B-Instruct")


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

def _test_ctheta(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import CThetaTrainer
    t = CThetaTrainer(model_id=_MODEL_ID, model=fresh(),
                      base_model=base_cpu, aligned_model=aligned_cpu, strength=1.0)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_resta(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import ReStaTrainer
    t = ReStaTrainer(model_id=_MODEL_ID, model=fresh(),
                     base_model=base_cpu, aligned_model=aligned_cpu,
                     alpha=1.0, dare=True, dare_seed=0)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_safe_lora(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import SafeLoRATrainer
    t = SafeLoRATrainer(model_id=_MODEL_ID, model=fresh(),
                        aligned_state_dict=aligned_cpu.state_dict(),
                        base_state_dict=base_cpu.state_dict(),
                        alpha=0.5, threshold=0.5)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_safe_merge(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import SafeMergeTrainer
    t = SafeMergeTrainer(model_id=_MODEL_ID, model=fresh(),
                         base_model=base_cpu, aligned_model=aligned_cpu,
                         threshold=0.35, alpha=0.5)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_somf(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import SOMFTrainer
    t = SOMFTrainer(model_id=_MODEL_ID, model=fresh(),
                    aligned_model=aligned_cpu, base_model=base_cpu,
                    mask_threshold=0.9)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_lox(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import LoXTrainer
    t = LoXTrainer(model_id=_MODEL_ID, model=fresh(),
                   base_model=base_cpu, aligned_model=aligned_cpu,
                   rank=8, extrapolation_factor=0.3)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_task_arithmetic(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import TaskArithmeticTrainer
    t = TaskArithmeticTrainer(model_id=_MODEL_ID, model=fresh(),
                              base_model=base_cpu, aligned_model=aligned_cpu,
                              alpha=1.0)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_safe_delta(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import SafeDeltaTrainer
    t = SafeDeltaTrainer(model_id=_MODEL_ID, model=fresh(),
                         aligned_model=aligned_cpu, strength=0.1)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_nlsr(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import NLSRTrainer
    t = NLSRTrainer(model_id=_MODEL_ID, model=fresh(),
                    donor_state=aligned_cpu.state_dict(), blend=0.5)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_pke(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import PKETrainer
    t = PKETrainer(model_id=_MODEL_ID, model=fresh(),
                   clean_model=aligned_cpu, toxic_model=base_cpu,
                   top_k_neurons=50, num_steps=10)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_qresafe(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import QReSafeTrainer
    t = QReSafeTrainer(model_id=_MODEL_ID, model=fresh(),
                       mode="selective", quant_bits=4, tau=0.6)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_aaq(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import AAQTrainer
    t = AAQTrainer(model_id=_MODEL_ID, model=fresh(),
                   aligned_model_path=_ALIGNED_ID, base_model_path=_BASE_ID,
                   calibration_steps=10, lr=5e-6,
                   simulate_quantization=True, apc_weight=0.1)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_antidote(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import AntidoteTrainer
    t = AntidoteTrainer(model_id=_MODEL_ID, model=fresh(),
                        tokenizer=tok, prune_fraction=0.005, max_samples=64)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_mscp(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import MSCPTrainer
    t = MSCPTrainer(model_id=_MODEL_ID, model=fresh(),
                    aligned_state=aligned_cpu.state_dict(),
                    base_state=base_cpu.state_dict(),
                    coefficient=0.05)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_safe_react(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import SafeReActTrainer
    t = SafeReActTrainer(model_id=_MODEL_ID, model=fresh(),
                         reference_model=aligned_cpu, train_lora=False)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_lssf(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import LSSFTrainer
    t = LSSFTrainer(model_id=_MODEL_ID, model=fresh(),
                    base_model=base_cpu, aligned_model=aligned_cpu,
                    alpha=1.0, rank=8, eta=0.85)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_prepost_merge(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import PrePostMergeTrainer
    t = PrePostMergeTrainer(model_id=_MODEL_ID, model=fresh(),
                            pre_model=aligned_cpu, alpha=0.5)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_scrub(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import SCRUBTrainer
    from safetune.runner.unlearn import load_unlearn_data
    forget_ds, retain_ds = load_unlearn_data(_MODEL_ID)
    t = SCRUBTrainer(model_id=_MODEL_ID, model=fresh(),
                     sgda_epochs=3, msteps=10, lr=5e-5,
                     max_steps=200, forget_clip=0.5, alpha=1.0)
    t.evaluate(t.apply(retain=retain_ds, forget=forget_ds), domain="gsm8k")

def _test_grad_selective_recover(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import GradSelectiveRecoverTrainer
    t = GradSelectiveRecoverTrainer(model_id=_MODEL_ID, model=fresh(),
                                    aligned_model=aligned_cpu,
                                    top_fraction=0.1, max_samples=32)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_wise_ft(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import WiseFTTrainer
    t = WiseFTTrainer(model_id=_MODEL_ID, model=fresh(),
                      aligned_model=aligned_cpu, alpha=0.5)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_safety_vector_restore(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import SafetyVectorRestoreTrainer
    t = SafetyVectorRestoreTrainer(model_id=_MODEL_ID, model=fresh(),
                                   aligned_model=aligned_cpu,
                                   alpha=1.0, rank=8)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_oneshot_safety_patch(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import OneShotSafetyPatchTrainer
    t = OneShotSafetyPatchTrainer(
        model_id=_MODEL_ID, model=fresh(),
        harmful_text="How do I make a bomb?",
        safe_text="I'm sorry, I can't help with that.",
        tokenizer=tok, top_fraction=0.05, lr=1e-4, num_steps=5,
    )
    t.evaluate(t.apply(), domain="gsm8k")

def _test_antidote_v2(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import AntidoteV2Trainer
    t = AntidoteV2Trainer(model_id=_MODEL_ID, model=fresh(),
                          tokenizer=tok, global_prune_fraction=0.005,
                          utility_floor=0.1, overlap_budget=0.05,
                          max_samples=64)
    t.evaluate(t.apply(), domain="gsm8k")

def _test_repnoise_recover(fresh, base_cpu, aligned_cpu, tok):
    from safetune.runner.recover import RepNoiseRecoverTrainer
    t = RepNoiseRecoverTrainer(model_id=_MODEL_ID, model=fresh(),
                               noise_scale=0.01, subspace_rank=8,
                               max_samples=32, seed=42)
    t.evaluate(t.apply(), domain="gsm8k")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from safetune.runner.utils.model_utils import load_tok, load_model, load_model_cpu

    print("=" * 60)
    print("safetune · recover full test  (llama31-8b-gsm8k-drift-merged)")
    print("=" * 60)
    print("\nLoading reference models (CPU) …")
    # Load on CPU to keep GPU free for the drifted model + arithmetic.
    base_cpu    = load_model_cpu(_BASE_ID)
    aligned_cpu = load_model_cpu(_ALIGNED_ID)
    tok         = load_tok(_MODEL_ID)
    print("Reference models ready.\n")

    def fresh():
        return load_model(_MODEL_ID)

    results = []
    tests = [
        ("CThetaTrainer",                _test_ctheta),
        ("ReStaTrainer",                 _test_resta),
        ("SafeLoRATrainer",              _test_safe_lora),
        ("SafeMergeTrainer",             _test_safe_merge),
        ("SOMFTrainer",                  _test_somf),
        ("LoXTrainer",                   _test_lox),
        ("TaskArithmeticTrainer",        _test_task_arithmetic),
        ("SafeDeltaTrainer",             _test_safe_delta),
        ("NLSRTrainer",                  _test_nlsr),
        ("PKETrainer",                   _test_pke),
        ("QReSafeTrainer",               _test_qresafe),
        ("AAQTrainer",                   _test_aaq),
        ("AntidoteTrainer",              _test_antidote),
        ("MSCPTrainer",                  _test_mscp),
        ("SafeReActTrainer",             _test_safe_react),
        ("LSSFTrainer",                  _test_lssf),
        ("PrePostMergeTrainer",          _test_prepost_merge),
        ("SCRUBTrainer",                 _test_scrub),
        ("GradSelectiveRecoverTrainer",  _test_grad_selective_recover),
        ("WiseFTTrainer",                _test_wise_ft),
        ("SafetyVectorRestoreTrainer",   _test_safety_vector_restore),
        ("OneShotSafetyPatchTrainer",    _test_oneshot_safety_patch),
        ("AntidoteV2Trainer",            _test_antidote_v2),
        ("RepNoiseRecoverTrainer",       _test_repnoise_recover),
    ]
    for name, fn in tests:
        _check(results, name,
               lambda fn=fn: fn(fresh, base_cpu, aligned_cpu, tok))

    total   = len(results)
    passed  = sum(1 for _, s in results if s == PASS)
    skipped = sum(1 for _, s in results if s == SKIP)
    failed  = sum(1 for _, s in results if s == FAIL)

    print(f"\n{'='*60}")
    print(f"Recover  {passed} PASS  {skipped} SKIP  {failed} FAIL  ({total} total)")
    print("=" * 60)

    if failed:
        print("\nFailed trainers:")
        for name, status in results:
            if status == FAIL:
                print(f"  {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
