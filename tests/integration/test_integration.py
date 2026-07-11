#!/usr/bin/env python3
"""SafeTune — four-pillar quick start.

All trainers are zero-arg instantiable.  Override defaults with kwargs.
Dataset loaders need no tokenizer — they use DEFAULT_MODEL_ID internally.

Usage:
    python examples/test.py
"""

import os

from safetune.runner.utils.model_utils import load_tok, load_model, load_model_cpu
from safetune.runner.utils.dataset import load_harden_dataset
from safetune.runner.harden import SafeGradTrainer
from safetune.runner.recover import ReStaTrainer
from safetune.runner.unlearn import NPOTrainer, load_unlearn_data
from safetune.runner.steer import CAATrainer, load_steer_data

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ── External-checkpoint requirement ──────────────────────────────────────────
# Each pillar demo runs on a *drifted* checkpoint (+ base / aligned refs for
# recover). These checkpoints are NOT shipped with the repo; provide them via
# env vars:
#
#   SAFETUNE_DRIFT_MODEL   — drifted model (required)
#   SAFETUNE_BASE_MODEL    — pre-alignment base (recover)
#   SAFETUNE_ALIGNED_MODEL — aligned reference (recover)
#
_MODEL_ID   = os.environ.get("SAFETUNE_DRIFT_MODEL", "")
_BASE_ID    = os.environ.get("SAFETUNE_BASE_MODEL", "")
_ALIGNED_ID = os.environ.get("SAFETUNE_ALIGNED_MODEL", "")


def test_harden():
    tokenizer = load_tok(_MODEL_ID)
    model = load_model(_MODEL_ID)

    train_set, safety_set = load_harden_dataset(tokenizer)

    trainer = SafeGradTrainer(
        model_id=_MODEL_ID,
        model=model,
        tokenizer=tokenizer,
        rho=1.0,
        kl_temperature=1.0,
    )

    out_path = trainer.train(train_set, out_dir="safegrad_run", safety_dataset=safety_set)
    print(out_path)
    trainer.evaluate(out_path, domain="gsm8k")
    print(train_set)


def test_resta():
    model = load_model(_MODEL_ID)
    # Load base/aligned on CPU — task arithmetic is state-dict arithmetic,
    # no forward passes needed, keeping them off GPU saves ~32 GiB.
    base_model    = load_model_cpu(_BASE_ID)
    aligned_model = load_model_cpu(_ALIGNED_ID)

    trainer = ReStaTrainer(
        model_id=_MODEL_ID,
        model=model,
        base_model=base_model,
        aligned_model=aligned_model,
        # alpha=1.0 with DARE (drop 0.9, ×10 rescale) over-perturbs the weights
        # and destroys the LM (wikitext ppl ~160k, gsm8k 0). alpha=0.5 restores
        # safety while keeping the model usable; dare_seed pins the DARE mask so
        # the run is reproducible.
        alpha=0.5,
        dare=True,
        dare_seed=0,
    )

    patched = trainer.apply()
    trainer.evaluate(patched, domain="gsm8k")


def test_npo():
    model = load_model(_MODEL_ID)
    forget_ds, retain_ds = load_unlearn_data(_MODEL_ID)

    trainer = NPOTrainer(
        model_id=_MODEL_ID,
        model=model,
        variant="npo_grad_diff",
        beta=0.1,
        num_epochs=5,
        lr=1e-5,
    )

    unlearned = trainer.unlearn(forget_ds, retain_ds)
    trainer.evaluate(unlearned, domain="gsm8k")


def test_caa():
    tokenizer = load_tok(_MODEL_ID)
    model = load_model(_MODEL_ID)
    harmful, harmless = load_steer_data(256)

    trainer = CAATrainer(
        model_id=_MODEL_ID,
        model=model,
        tokenizer=tokenizer,
        multiplier=20.0,
    )

    caa_model, _ = trainer.calibrate(harmful, harmless)
    trainer.evaluate(caa_model, domain="gsm8k")


if __name__ == '__main__':
    test_harden()
    test_resta()
    test_npo()
    test_caa()
