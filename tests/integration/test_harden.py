#!/usr/bin/env python3
"""Harden pillar — full end-to-end test for every trainer.

Each test runs .train() followed by .evaluate() on llama31-8b-gsm8k-drift-merged
and reports PASS / FAIL / SKIP.

Usage::

    python examples/test_harden.py
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
#   SAFETUNE_ALIGNED_MODEL — path/HF-id of the aligned reference
#
# Reference checkpoints were produced from real drifted fine-tunes.

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MERGED     = os.path.abspath(os.path.join(_SCRIPT_DIR, "../audit/training/merged"))
_MODEL_ID   = os.environ.get("SAFETUNE_DRIFT_MODEL",
                             os.path.join(_MERGED, "llama31-8b-gsm8k-drift-merged"))
_ALIGNED_ID = os.environ.get("SAFETUNE_ALIGNED_MODEL",
                             os.path.join(_MERGED, "llama31-8b-hh-rlhf-aligned-final"))

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

def _test_plain_sft(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import PlainSFTTrainer
    t = PlainSFTTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok())
    out = t.train(train_ds, out_dir="plainsft_run")
    t.evaluate(out, domain="gsm8k")

def _test_safegrad(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import SafeGradTrainer
    t = SafeGradTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok(),
                        rho=1.0, kl_temperature=1.0)
    out = t.train(train_ds, out_dir="safegrad_run", safety_dataset=safety_ds)
    t.evaluate(out, domain="gsm8k")

def _test_lisa(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import LisaTrainer
    t = LisaTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok(),
                    lisa_rho=0.1, lisa_warmup_steps=10,
                    lisa_alignment_step=20, lisa_finetune_step=20)
    out = t.train(train_ds, out_dir="lisa_run", safety_dataset=safety_ds)
    t.evaluate(out, domain="gsm8k")

def _test_sppft(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import SPPFTTrainer
    t = SPPFTTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok())
    out = t.train(train_ds, out_dir="sppft_run")
    t.evaluate(out, domain="gsm8k")

def _test_lookahead(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import LookAheadTrainer
    t = LookAheadTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok(),
                         prefix_mode="virtual", prefix_length=6)
    out = t.train(train_ds, out_dir="lookahead_run")
    t.evaluate(out, domain="gsm8k")

def _test_stardss(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import STARDSSTrainer
    t = STARDSSTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok(),
                       use_kl_penalty=True, kl_scale=1.0)
    out = t.train(train_ds, out_dir="stardss_run")
    t.evaluate(out, domain="gsm8k")

def _test_sap(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import SAPTrainer
    t = SAPTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok(),
                   grad_rate=0.1, v_update_step=0.05, contrastive_temperature=1.0)
    out = t.train(train_ds, out_dir="sap_run", safety_dataset=safety_ds)
    t.evaluate(out, domain="gsm8k")

def _test_asft(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import AsFTTrainer
    t = AsFTTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok(),
                    reg_lambda=1.0, aligned_model_path=_ALIGNED_ID)
    out = t.train(train_ds, out_dir="asft_run")
    t.evaluate(out, domain="gsm8k")

def _test_surgery(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import SurgeryTrainer
    t = SurgeryTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok(),
                       sink_lambda=0.01)
    out = t.train(train_ds, out_dir="surgery_run", safety_dataset=safety_ds)
    t.evaluate(out, domain="gsm8k")

def _test_derta(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import DeRTaTrainer
    t = DeRTaTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok())
    out = t.train(train_ds, out_dir="derta_run")
    t.evaluate(out, domain="gsm8k")

def _test_door(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import DOORTrainer
    t = DOORTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok(),
                    beta=0.5, refusal_w=1.0, unlearn_w=1.0,
                    base_model_path=_MODEL_ID)
    out = t.train(train_ds, out_dir="door_run")
    t.evaluate(out, domain="gsm8k")

def _test_tar(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import TARTrainer
    t = TARTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok(),
                   inner_steps=25, inner_lr=1e-4, lambda_tar=1.0)
    out = t.train(train_ds, out_dir="tar_run")
    t.evaluate(out, domain="gsm8k")

def _test_salora(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import SaLoRATrainer
    t = SaLoRATrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok(),
                      rank=16, safety_rank=16, strength=1.0)
    out = t.train(train_ds, out_dir="salora_run", safety_dataset=safety_ds)
    t.evaluate(out, domain="gsm8k")

def _test_vaccine(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import VaccineTrainer
    t = VaccineTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok(), rho=2.0)
    out = t.train(train_ds, out_dir="vaccine_run")
    t.evaluate(out, domain="gsm8k")

def _test_tvaccine(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import TVaccineTrainer
    t = TVaccineTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok(),
                        rho=2.0, top_k_ratio=0.5)
    out = t.train(train_ds, out_dir="tvaccine_run")
    t.evaluate(out, domain="gsm8k")

def _test_booster(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import BoosterTrainer
    t = BoosterTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok(),
                       perturb_scale=0.01)
    out = t.train(train_ds, out_dir="booster_run")
    t.evaluate(out, domain="gsm8k")

def _test_repnoise(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import RepNoiseTrainer
    t = RepNoiseTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok(),
                        noise_alpha=1.0, noise_beta=0.1, noise_gamma=1.0)
    out = t.train(train_ds, out_dir="repnoise_run", safety_dataset=safety_ds)
    t.evaluate(out, domain="gsm8k")

def _test_ctrap(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import CTRAPTrainer
    t = CTRAPTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok())
    out = t.train(train_ds, out_dir="ctrap_run", safety_dataset=safety_ds)
    t.evaluate(out, domain="gsm8k")

def _test_seam(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import SEAMTrainer
    t = SEAMTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok())
    out = t.train(train_ds, out_dir="seam_run", safety_dataset=safety_ds)
    t.evaluate(out, domain="gsm8k")

def _test_seal(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import SEALTrainer
    t = SEALTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok())
    out = t.train(train_ds, out_dir="seal_run", safety_dataset=safety_ds)
    t.evaluate(out, domain="gsm8k")

def _test_constrained_sft(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import ConstrainedSFTTrainer
    t = ConstrainedSFTTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok())
    out = t.train(train_ds, out_dir="constrained_sft_run")
    t.evaluate(out, domain="gsm8k")

def _test_lox_harden(fresh, tok, train_ds, safety_ds):
    from safetune.runner.harden import LoXHardenTrainer
    t = LoXHardenTrainer(model_id=_MODEL_ID, model=fresh(), tokenizer=tok(),
                         rank=8, extrapolation_factor=0.3,
                         aligned_model_path=_ALIGNED_ID)
    out = t.train(train_ds, out_dir="lox_harden_run")
    t.evaluate(out, domain="gsm8k")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from safetune.runner.harden import load_harden_data
    from safetune.runner.utils.model_utils import load_tok, load_model

    print("=" * 60)
    print("safetune · harden full test  (llama31-8b-gsm8k-drift-merged)")
    print("=" * 60)
    print("\nLoading datasets …")
    train_ds, safety_ds = load_harden_data(_MODEL_ID)
    print("Datasets ready.\n")

    def fresh():
        return load_model(_MODEL_ID)

    def tok():
        return load_tok(_MODEL_ID)

    results = []
    tests = [
        ("PlainSFTTrainer",       _test_plain_sft),
        ("SafeGradTrainer",       _test_safegrad),
        ("LisaTrainer",           _test_lisa),
        ("SPPFTTrainer",          _test_sppft),
        ("LookAheadTrainer",      _test_lookahead),
        ("STARDSSTrainer",        _test_stardss),
        ("SAPTrainer",            _test_sap),
        ("AsFTTrainer",           _test_asft),
        ("SurgeryTrainer",        _test_surgery),
        ("DeRTaTrainer",          _test_derta),
        ("DOORTrainer",           _test_door),
        ("TARTrainer",            _test_tar),
        ("SaLoRATrainer",         _test_salora),
        ("VaccineTrainer",        _test_vaccine),
        ("TVaccineTrainer",       _test_tvaccine),
        ("BoosterTrainer",        _test_booster),
        ("RepNoiseTrainer",       _test_repnoise),
        ("CTRAPTrainer",          _test_ctrap),
        ("SEAMTrainer",           _test_seam),
        ("SEALTrainer",           _test_seal),
        ("ConstrainedSFTTrainer", _test_constrained_sft),
        ("LoXHardenTrainer",      _test_lox_harden),
    ]
    for name, fn in tests:
        _check(results, name, lambda fn=fn: fn(fresh, tok, train_ds, safety_ds))

    total   = len(results)
    passed  = sum(1 for _, s in results if s == PASS)
    skipped = sum(1 for _, s in results if s == SKIP)
    failed  = sum(1 for _, s in results if s == FAIL)

    print(f"\n{'='*60}")
    print(f"Harden  {passed} PASS  {skipped} SKIP  {failed} FAIL  ({total} total)")
    print("=" * 60)

    if failed:
        print("\nFailed trainers:")
        for name, status in results:
            if status == FAIL:
                print(f"  {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
