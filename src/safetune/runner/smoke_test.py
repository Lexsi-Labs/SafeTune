"""Smoke test — verify all runner trainer classes are importable and instantiable.

Runs without a GPU, without real datasets, and without executing any training.
Each trainer is instantiated with a tiny dummy model and a no-op tokenizer stub,
then inspected for required attributes and methods.

Usage::

    python -m safetune.runner.smoke_test
    # or
    python src/safetune/runner/smoke_test.py
"""

from __future__ import annotations
import sys
import traceback
from typing import Any


# ── Tiny stub model and tokenizer ─────────────────────────────────────────────

class _StubTokenizer:
    name_or_path = "stub"
    pad_token = "<pad>"
    eos_token = "</s>"
    padding_side = "right"

    def __call__(self, *args, **kwargs):
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    def apply_chat_template(self, messages, **kwargs):
        return " ".join(m.get("content", "") for m in messages)


class _StubModel:
    device = "cpu"

    def parameters(self):
        import torch
        yield torch.zeros(1, requires_grad=True)

    def named_parameters(self):
        import torch
        yield "weight", torch.zeros(1, requires_grad=True)

    def state_dict(self):
        return {}

    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, *args, **kwargs):
        import torch

        class _Out:
            loss = torch.tensor(0.1, requires_grad=True)
            logits = torch.zeros(1, 10, 32000)

        return _Out()


_stub_model = _StubModel()
_stub_tok = _StubTokenizer()


# ── Test helpers ──────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

_results: list[tuple[str, str, str]] = []


def _check(pillar: str, cls_name: str, fn):
    try:
        fn()
        _results.append((pillar, cls_name, PASS))
        print(f"  [{PASS}] {pillar}.{cls_name}")
    except ImportError as e:
        # Optional deps (peft, lm_eval, vllm) not installed → expected SKIP
        _results.append((pillar, cls_name, SKIP))
        print(f"  [{SKIP}] {pillar}.{cls_name}: {e}")
    except Exception as e:
        _results.append((pillar, cls_name, FAIL))
        print(f"  [{FAIL}] {pillar}.{cls_name}: {e}")
        traceback.print_exc()


def _required_methods(obj, methods: list[str]):
    for m in methods:
        assert hasattr(obj, m), f"missing method: {m}"


# ── Harden smoke tests ────────────────────────────────────────────────────────

def _test_harden():
    from safetune.runner import harden

    HARDEN_CLASSES = [
        # (class_name, extra_kwargs)
        ("PlainSFTTrainer",        {}),
        ("SafeGradTrainer",        {"rho": 1.0}),
        ("LisaTrainer",            {"lisa_rho": 0.1}),
        ("SPPFTTrainer",           {}),
        ("LookAheadTrainer",       {"prefix_mode": "virtual", "prefix_length": 6}),
        ("STARDSSTrainer",         {"use_kl_penalty": True}),
        ("SAPTrainer",             {"grad_rate": 0.1}),
        ("AsFTTrainer",            {"reg_lambda": 1.0}),
        ("SurgeryTrainer",         {"sink_lambda": 0.01}),
        ("DeRTaTrainer",           {}),
        ("DOORTrainer",            {"beta": 0.5}),
        ("TARTrainer",             {"inner_steps": 25}),
        ("SaLoRATrainer",          {"rank": 16, "safety_rank": 16}),
        ("VaccineTrainer",         {"rho": 2.0}),
        ("TVaccineTrainer",        {"rho": 2.0, "top_k_ratio": 0.5}),
        ("BoosterTrainer",         {"perturb_scale": 0.01}),
        ("RepNoiseTrainer",        {"noise_alpha": 1.0}),
        ("CTRAPTrainer",           {}),
        ("SEAMTrainer",            {}),
        ("SEALTrainer",            {}),
        ("ConstrainedSFTTrainer",  {}),
        ("LoXHardenTrainer",       {"rank": 8}),
    ]

    for cls_name, kwargs in HARDEN_CLASSES:
        def _fn(cn=cls_name, kw=kwargs):
            cls = getattr(harden, cn)
            obj = cls(_stub_model, _stub_tok, **kw)
            _required_methods(obj, ["train", "eval", "save_results"])
            assert obj.METHOD, f"{cn}.METHOD is empty"
        _check("harden", cls_name, _fn)


# ── Recover smoke tests ───────────────────────────────────────────────────────

def _test_recover():
    from safetune.runner import recover

    RECOVER_CLASSES = [
        ("CThetaTrainer",                  {"strength": 1.0}),
        ("ReStaTrainer",                   {"alpha": 1.0}),
        ("SafeLoRATrainer",                {"alpha": 0.5, "threshold": 0.5}),
        ("SafeMergeTrainer",               {"threshold": 0.35, "alpha": 0.5}),
        ("SOMFTrainer",                    {"mask_threshold": 0.9}),
        ("LoXTrainer",                     {"rank": 8, "extrapolation_factor": 0.3}),
        ("TaskArithmeticTrainer",          {"alpha": 1.0}),
        ("SafeDeltaTrainer",               {"strength": 0.1}),
        ("NLSRTrainer",                    {"blend": 0.5}),
        ("PKETrainer",                     {"top_k_neurons": 50, "num_steps": 10}),
        ("QReSafeTrainer",                 {"mode": "selective", "quant_bits": 4}),
        ("AAQTrainer",                     {"calibration_steps": 10, "lr": 5e-6}),
        ("AntidoteTrainer",                {"prune_fraction": 0.005}),
        ("MSCPTrainer",                    {"coefficient": 0.05}),
        ("SafeReActTrainer",               {"train_lora": False}),
        ("LSSFTrainer",                    {"alpha": 1.0, "rank": 8, "eta": 0.85}),
        ("PrePostMergeTrainer",            {"alpha": 0.5}),
        ("SCRUBTrainer",                   {"sgda_epochs": 3, "msteps": 10}),
        ("GradSelectiveRecoverTrainer",    {"top_fraction": 0.1}),
        ("WiseFTTrainer",                  {"alpha": 0.5}),
        ("SafetyVectorRestoreTrainer",     {"alpha": 1.0, "rank": 8}),
        ("OneShotSafetyPatchTrainer",      {"top_fraction": 0.05}),
        ("AntidoteV2Trainer",              {"global_prune_fraction": 0.005}),
        ("RepNoiseRecoverTrainer",         {"noise_scale": 0.01}),
    ]

    for cls_name, kwargs in RECOVER_CLASSES:
        def _fn(cn=cls_name, kw=kwargs):
            cls = getattr(recover, cn)
            obj = cls(_stub_model, **kw)
            _required_methods(obj, ["apply", "eval", "save_results", "save_checkpoint"])
            assert obj.METHOD, f"{cn}.METHOD is empty"
        _check("recover", cls_name, _fn)


# ── Steer smoke tests ─────────────────────────────────────────────────────────

def _test_steer():
    from safetune.runner import steer

    STEER_CLASSES = [
        ("RefusalDirectionTrainer",      {"alpha": 20.0}),
        ("CAATrainer",                   {"multiplier": 20.0}),
        ("LinearProbeGuardTrainer",      {"layer": 15}),
        ("SCANSTrainer",                 {}),
        ("STATrainer",                   {}),
        ("CircuitBreakerRRTrainer",      {"threshold": 0.5}),
        ("CASTTrainer",                  {"probe_layer": 15, "alpha": 10.0}),
        ("AdaSteerTrainer",              {"alpha": 15.0}),
        ("SafeSwitchTrainer",            {"gate_layer": 16}),
        ("SafeDecodingTrainer",          {"alpha": 2.0}),
        ("ContrastiveDecodingTrainer",   {"alpha": 1.0}),
        ("ProxyTuningTrainer",           {"alpha": 1.0}),
        ("NudgingTrainer",               {"nudge_strength": 5.0}),
        ("CircuitBreakerTrainer",        {}),
        ("RepBendTrainer",               {}),
        ("TARSteerTrainer",              {}),
        ("RRFAEnsembleTrainer",          {}),
        ("SafeSteerTrainer",             {"alpha": 15.0}),
        ("AlphaSteerTrainer",            {"alpha": 20.0}),
    ]

    for cls_name, kwargs in STEER_CLASSES:
        def _fn(cn=cls_name, kw=kwargs):
            cls = getattr(steer, cn)
            obj = cls(_stub_model, _stub_tok, **kw)
            _required_methods(obj, ["calibrate", "save_results"])
            assert obj.METHOD, f"{cn}.METHOD is empty"
        _check("steer", cls_name, _fn)


# ── Unlearn smoke tests ───────────────────────────────────────────────────────

def _test_unlearn():
    from safetune.runner import unlearn

    UNLEARN_CLASSES = [
        ("RMUTrainer",              {"layer_id": 7, "max_num_batches": 80}),
        ("NPOTrainer",              {"variant": "npo_grad_diff", "beta": 0.1}),
        ("GradientAscentTrainer",   {"forget_loss": "grad_ascent", "epochs": 5}),
        ("GradDiffTrainer",         {"epochs": 5}),
        ("FLATTrainer",             {"variant": "flat_retain"}),
        ("SimDPOTrainer",           {"beta": 0.1}),
    ]

    for cls_name, kwargs in UNLEARN_CLASSES:
        def _fn(cn=cls_name, kw=kwargs):
            cls = getattr(unlearn, cn)
            obj = cls(_stub_model, **kw)
            _required_methods(obj, ["unlearn", "eval", "save_results", "save_checkpoint"])
            assert obj.METHOD, f"{cn}.METHOD is empty"
        _check("unlearn", cls_name, _fn)


# ── Utils smoke tests ─────────────────────────────────────────────────────────

def _test_utils():
    def _check_imports():
        from safetune.runner.utils import (
            SAFETY_BENCHES, UTILITY_TASKS_CORE, UTILITY_TASKS_BY_DRIFT,
        )
        assert len(SAFETY_BENCHES) == 7, f"expected 7 safety benches, got {len(SAFETY_BENCHES)}"
        assert "gsm8k" in UTILITY_TASKS_CORE
        assert "code" in UTILITY_TASKS_BY_DRIFT
        from safetune.runner.utils.results_writer import DEFAULT_RESULTS_DIR, ResultsWriter
        import os
        assert DEFAULT_RESULTS_DIR.endswith("results"), f"unexpected results dir: {DEFAULT_RESULTS_DIR}"
        # The results dir is a runtime artifact (gitignored), created on first
        # write by ResultsWriter. Ensure it can be created here so a fresh
        # clone passes; this validates the path is wired and writable.
        os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
        assert os.path.isdir(DEFAULT_RESULTS_DIR), f"results dir not creatable: {DEFAULT_RESULTS_DIR}"

    _check("utils", "constants_and_results_dir", _check_imports)

    def _check_refusal_pairs():
        from safetune.runner.utils.data_utils import refusal_prompt_pairs
        harmful, harmless = refusal_prompt_pairs()
        assert len(harmful) == len(harmless) == 6

    _check("utils", "refusal_prompt_pairs", _check_refusal_pairs)

    def _check_runner_import():
        from safetune.runner import harden, recover, steer, unlearn
        assert hasattr(harden, "LisaTrainer")
        assert hasattr(recover, "CThetaTrainer")
        assert hasattr(steer, "RefusalDirectionTrainer")
        assert hasattr(unlearn, "RMUTrainer")

    _check("utils", "top_level_imports", _check_runner_import)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("safetune.runner smoke test")
    print("=" * 60)

    print("\n[utils]")
    _test_utils()

    print("\n[harden]")
    _test_harden()

    print("\n[recover]")
    _test_recover()

    print("\n[steer]")
    _test_steer()

    print("\n[unlearn]")
    _test_unlearn()

    # Summary
    total = len(_results)
    passed = sum(1 for _, _, s in _results if s == PASS)
    skipped = sum(1 for _, _, s in _results if s == SKIP)
    failed = sum(1 for _, _, s in _results if s == FAIL)

    print("\n" + "=" * 60)
    print(f"Results: {passed} PASS  {skipped} SKIP  {failed} FAIL  ({total} total)")
    print("=" * 60)

    if failed:
        print("\nFailed:")
        for pillar, cls_name, status in _results:
            if status == FAIL:
                print(f"  {pillar}.{cls_name}")
        sys.exit(1)
    else:
        print("All checks passed (or skipped due to optional deps).")
        sys.exit(0)


if __name__ == "__main__":
    main()
