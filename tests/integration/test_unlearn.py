#!/usr/bin/env python3
"""Unlearn pillar — full end-to-end test for every trainer.

Each test runs .unlearn() followed by .evaluate() on llama31-8b-gsm8k-drift-merged
and reports PASS / FAIL / SKIP.

Usage::

    python examples/test_unlearn.py
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
#   SAFETUNE_DRIFT_MODEL — path/HF-id of the drifted (gsm8k) model to unlearn on
#
# The forget/retain data IS public and downloads automatically.

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MERGED     = os.path.abspath(os.path.join(_SCRIPT_DIR, "../audit/training/merged"))
_MODEL_ID   = os.environ.get("SAFETUNE_DRIFT_MODEL",
                             os.path.join(_MERGED, "llama31-8b-gsm8k-drift-merged"))

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

def _test_rmu(fresh, forget_ds, retain_ds, tok):
    from safetune.runner.unlearn import RMUTrainer
    t = RMUTrainer(model_id=_MODEL_ID, model=fresh(),
                   layer_id=7, update_layer_ids=[5, 6, 7],
                   max_num_batches=80, lr=5e-5,
                   alpha=1200.0, steering_coeff=20.0)
    t.evaluate(t.unlearn(forget_ds, retain_ds), domain="gsm8k")

def _test_npo(fresh, forget_ds, retain_ds, tok):
    from safetune.runner.unlearn import NPOTrainer
    t = NPOTrainer(model_id=_MODEL_ID, model=fresh(),
                   variant="npo_grad_diff", beta=0.1,
                   num_epochs=5, lr=1e-5)
    t.evaluate(t.unlearn(forget_ds, retain_ds), domain="gsm8k")

def _test_gradient_ascent(fresh, forget_ds, retain_ds, tok):
    from safetune.runner.unlearn import GradientAscentTrainer
    t = GradientAscentTrainer(model_id=_MODEL_ID, model=fresh(),
                              forget_loss="grad_ascent",
                              epochs=5, max_steps=200,
                              lr=1e-5, forget_clip=0.5)
    t.evaluate(t.unlearn(forget_ds, retain_ds), domain="gsm8k")

def _test_grad_diff(fresh, forget_ds, retain_ds, tok):
    from safetune.runner.unlearn import GradDiffTrainer
    t = GradDiffTrainer(model_id=_MODEL_ID, model=fresh(),
                        epochs=5, max_steps=200,
                        lr=1e-5, forget_clip=0.5)
    t.evaluate(t.unlearn(forget_ds, retain_ds), domain="gsm8k")

def _test_flat(fresh, forget_ds, retain_ds, tok):
    from safetune.runner.unlearn import FLATTrainer
    t = FLATTrainer(model_id=_MODEL_ID, model=fresh(),
                    variant="flat_retain", epochs=5,
                    lr=1e-5, forget_clip=0.5, retain_coeff=1.0)
    t.evaluate(t.unlearn(forget_ds, retain_ds), domain="gsm8k")

def _test_simdpo(fresh, forget_ds, retain_ds, tok):
    from safetune.runner.unlearn import SimDPOTrainer
    t = SimDPOTrainer(model_id=_MODEL_ID, model=fresh(),
                      variant="simdpo_retain", beta=0.1,
                      epochs=5, lr=1e-5, retain_coeff=1.0)
    forget_pairs = t.make_simdpo_pairs(forget_ds, tok)
    t.evaluate(t.unlearn(forget_pairs, retain_ds), domain="gsm8k")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from safetune.runner.unlearn import load_unlearn_data
    from safetune.runner.utils.model_utils import load_tok, load_model

    print("=" * 60)
    print("safetune · unlearn full test  (llama31-8b-gsm8k-drift-merged)")
    print("=" * 60)
    print("\nLoading datasets …")
    forget_ds, retain_ds = load_unlearn_data(_MODEL_ID)
    tok = load_tok(_MODEL_ID)
    print("Datasets ready.\n")

    def fresh():
        return load_model(_MODEL_ID)

    results = []
    tests = [
        ("RMUTrainer",              _test_rmu),
        ("NPOTrainer",              _test_npo),
        ("GradientAscentTrainer",   _test_gradient_ascent),
        ("GradDiffTrainer",         _test_grad_diff),
        ("FLATTrainer",             _test_flat),
        ("SimDPOTrainer",           _test_simdpo),
    ]
    for name, fn in tests:
        _check(results, name,
               lambda fn=fn: fn(fresh, forget_ds, retain_ds, tok))

    total   = len(results)
    passed  = sum(1 for _, s in results if s == PASS)
    skipped = sum(1 for _, s in results if s == SKIP)
    failed  = sum(1 for _, s in results if s == FAIL)

    print(f"\n{'='*60}")
    print(f"Unlearn  {passed} PASS  {skipped} SKIP  {failed} FAIL  ({total} total)")
    print("=" * 60)

    if failed:
        print("\nFailed trainers:")
        for name, status in results:
            if status == FAIL:
                print(f"  {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
