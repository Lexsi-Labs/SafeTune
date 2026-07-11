"""Regression tests for bugs found during the CUDA recheck of PR #31.

Kept model-free / device-free where possible so they run on CPU, CUDA and MPS.

Bugs guarded here:
  1. `safetune train` (harden) never tokenized the raw dataset, so
     `_keep_model_columns` stripped every column and training died with
     ``num_samples=0``.
  2. `CSTTrainer(raw_examples=...)` handed a plain ``list`` to TRL ``DPOTrainer``
     (which calls ``.map``) → ``'list' object has no attribute 'map'``.
  3. AAQ "probe-free mode" built a *float* probe and fed it to a CausalLM's token
     embedding → ``RuntimeError: Expected ... Long ... got FloatTensor``.
  4. The eval demo invoked the abliteration attack with an invalid ``mode``.
"""
from pathlib import Path

import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# ── Bug 1: harden CLI must tokenize a raw dataset ─────────────────────────────
class TestHardenCliTokenizes:
    def test_passthrough_when_already_tokenized(self):
        from datasets import Dataset
        from safetune.cli import _ensure_tokenized

        ds = Dataset.from_list([{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]}])
        assert _ensure_tokenized(ds, tok=None) is ds  # tok unused when already tokenized

    def test_raises_without_a_prompt_column(self):
        from datasets import Dataset
        from safetune.cli import _ensure_tokenized

        ds = Dataset.from_list([{"foo": "bar", "baz": 1}])
        with pytest.raises(ValueError):
            _ensure_tokenized(ds, tok=None)

    def test_tokenizes_raw_prompt_response_rows(self, monkeypatch):
        from datasets import Dataset
        import safetune.runner.utils.data_utils as du
        from safetune.cli import _ensure_tokenized

        captured = {}

        def fake_tok_rows(tok, rows, max_len):
            captured["rows"] = rows
            return [{"input_ids": [0], "attention_mask": [1], "labels": [0]} for _ in rows]

        monkeypatch.setattr(du, "_tokenize_qa_rows", fake_tok_rows)
        ds = Dataset.from_list([
            {"prompt": "hi", "response": "hello", "category": "x", "is_safe": True},
            {"prompt": "yo", "response": "sup", "category": "y", "is_safe": False},
        ])
        out = _ensure_tokenized(ds, tok=object())
        assert out.num_rows == 2
        assert {"input_ids", "attention_mask", "labels"}.issubset(out.column_names)
        assert captured["rows"] == [("hi", "hello"), ("yo", "sup")]

    def test_kept_columns_are_non_empty_after_tokenization(self, monkeypatch):
        """The actual num_samples=0 guard: after tokenizing, _keep_model_columns
        must leave rows (removing ALL columns collapses a Dataset to 0 rows)."""
        from datasets import Dataset
        import safetune.runner.utils.data_utils as du
        from safetune.cli import _ensure_tokenized
        from safetune.runner.harden._base import _keep_model_columns

        monkeypatch.setattr(
            du, "_tokenize_qa_rows",
            lambda tok, rows, max_len: [
                {"input_ids": [0, 1], "attention_mask": [1, 1], "labels": [0, 1]} for _ in rows
            ],
        )
        raw = Dataset.from_list([{"prompt": "a", "response": "b"}] * 3)
        kept = _keep_model_columns(_ensure_tokenized(raw, tok=object()))
        assert kept.num_rows == 3

    def test_do_harden_wires_tokenization(self):
        import inspect
        from safetune import cli

        assert "_ensure_tokenized(_load_train_dataset" in inspect.getsource(cli._do_harden)


# ── Bug 2: CSTTrainer wraps raw_examples into a datasets.Dataset ───────────────
class TestCSTTrainerWrapsRawExamples:
    def test_raw_examples_become_a_dataset(self, monkeypatch):
        pytest.importorskip("trl")
        from datasets import Dataset
        import safetune.interventions.harden.cst as cstmod

        if getattr(cstmod, "_CST_IMPORT_ERROR", None) is not None:
            pytest.skip("CST formatter unavailable")

        captured = {}

        def fake_init(self, *a, **k):
            captured["train_dataset"] = k.get("train_dataset")

        monkeypatch.setattr(cstmod.DPOTrainer, "__init__", fake_init, raising=True)
        raw = [{"prompt": "p", "safe_response": "s", "unsafe_response": "u"}]
        cstmod.CSTTrainer(object(), raw_examples=raw)

        assert isinstance(captured["train_dataset"], Dataset)
        assert captured["train_dataset"].num_rows >= 1


# ── Bug 3: AAQ probe-free mode uses Long token ids for a vocab model ──────────
@pytest.mark.skipif(torch is None, reason="torch required")
class TestAAQProbeFreeProbe:
    def _model(self, vocab, param_last_dim=8):
        class Cfg:
            vocab_size = vocab

        class M:
            config = Cfg()

            def parameters(self):
                return iter([torch.zeros(param_last_dim, param_last_dim)])

        return M()

    def test_long_ids_for_vocab_model(self):
        from safetune.core.patches.aaq_patch import _make_probe_ids

        ids = _make_probe_ids(None, self._model(vocab=1000), torch.device("cpu"))
        assert ids.dtype in (torch.long, torch.int64, torch.int32, torch.int)
        assert int(ids.min()) >= 0 and int(ids.max()) < 1000

    def test_float_probe_for_vocabless_model(self):
        from safetune.core.patches.aaq_patch import _make_probe_ids

        out = _make_probe_ids(None, self._model(vocab=None), torch.device("cpu"))
        assert out.dtype == torch.float32


# ── Bug 5: eval backend auto-falls-back to transformers when vLLM is absent ───
class TestEvalBackendAutoFallback:
    def test_eval_functions_accept_backend(self):
        import inspect
        from safetune.runner.utils import eval_runner as er

        assert "backend" in inspect.signature(er.eval_safety).parameters
        assert "backend" in inspect.signature(er.eval_utility).parameters

    def test_safety_eval_is_not_hardcoded_to_vllm(self):
        import inspect
        from safetune.runner.utils import eval_runner as er

        src = inspect.getsource(er.eval_safety) + inspect.getsource(er._eval_advbench_inline)
        assert 'backend="vllm"' not in src

    def test_auto_falls_back_to_hf_without_vllm(self, monkeypatch):
        """The key fix: no vllm, no env var, no config → eval still works (hf)."""
        from safetune.runner.utils import eval_runner as er

        monkeypatch.setattr(er, "_fast_backend_ready", lambda: False)
        monkeypatch.setattr(er, "_EVAL_BACKEND", None)
        assert er._resolve_backend() == "hf"
        # explicitly asking for vllm without vllm also degrades, not crashes
        assert er._resolve_backend("vllm") == "hf"
        # an explicit hf request is honored
        assert er._resolve_backend("hf") == "hf"

    def test_auto_prefers_vllm_when_available(self, monkeypatch):
        from safetune.runner.utils import eval_runner as er

        monkeypatch.setattr(er, "_fast_backend_ready", lambda: True)
        monkeypatch.setattr(er, "_EVAL_BACKEND", None)
        assert er._resolve_backend() == "vllm"

    def test_set_eval_backend_overrides(self, monkeypatch):
        from safetune.runner.utils import eval_runner as er

        monkeypatch.setattr(er, "_fast_backend_ready", lambda: True)
        monkeypatch.setattr(er, "_EVAL_BACKEND", None)
        er.set_eval_backend("hf")
        try:
            assert er._resolve_backend() == "hf"
        finally:
            er.set_eval_backend(None)

    def test_backend_is_a_config_field(self):
        from safetune.config import SafeTuneConfig

        assert "eval_backend" in SafeTuneConfig.__dataclass_fields__
        assert SafeTuneConfig().eval_backend is None  # None → auto

    def test_steer_evaluate_forwards_backend(self):
        import inspect
        from safetune.runner.steer._base import _SteerBase

        assert "backend=backend" in inspect.getsource(_SteerBase.evaluate)

    def test_steer_eval_live_runs_the_utility_eval(self):
        """eval_live read utility_metrics() but never called eval_utility(), so
        the hf steer path always reported utility=N/A (issue #34)."""
        import inspect
        from safetune.runner.steer._base import _SteerBase

        src = inspect.getsource(_SteerBase.eval_live)
        assert "eval_utility(" in src
        # the eval_utility call must precede the utility_metrics read
        assert src.index("eval_utility(") < src.index("utility_metrics(")


# ── Bug 6: evaluate() must not crash when a metric mean is None ───────────────
class TestEvaluateToleratesMissingMetrics:
    def test_fmt_metric_handles_none(self):
        from safetune.runner.utils.eval_runner import fmt_metric

        assert fmt_metric(None) == "N/A"
        assert fmt_metric(0.7152) == "0.715"
        assert fmt_metric(1) == "1.000"

    def test_evaluate_methods_use_fmt_metric_not_raw_format(self):
        """harden/recover/unlearn used `{utility_mean(...):.3f}`, which raised
        TypeError when utility was None (e.g. capability backend unavailable)."""
        import inspect

        from safetune.runner.harden._base import _HardenBase
        from safetune.runner.recover._base import _RecoverBase
        from safetune.runner.unlearn._base import _UnlearnBase

        for cls in (_HardenBase, _RecoverBase, _UnlearnBase):
            src = inspect.getsource(cls.evaluate)
            assert "fmt_metric(" in src, cls.__name__
            assert "safety_mean(m):.3f" not in src, cls.__name__
            assert "utility_mean(m, drift_task=domain):.3f" not in src, cls.__name__


# ── Bug 7: RESTA (dare=False) mixed CPU/GPU tensors → device mismatch ─────────
@pytest.mark.skipif(torch is None, reason="torch required")
class TestRestaCrossDeviceApply:
    def _wrapper(self):
        from safetune.core.optim.resta import RESTAWrapper

        aligned = {"w": torch.ones(4)}       # kept on CPU (load_model_cpu)
        base = {"w": torch.zeros(4)}
        return RESTAWrapper(aligned_state_dict=aligned, base_state_dict=base)

    def test_apply_moves_safety_vector_to_finetuned_device(self):
        """RESTAWrapper.apply added a CPU safety vector to (possibly) GPU
        finetuned weights without a device move → RuntimeError on real GPUs."""
        wrapper = self._wrapper()
        ft = {"w": torch.full((4,), 2.0)}
        out = wrapper.apply(ft, alpha=1.0)
        assert out["w"].device == ft["w"].device
        assert torch.allclose(out["w"], torch.full((4,), 3.0))

    @pytest.mark.skipif(
        torch is None or not torch.cuda.is_available(), reason="CUDA required"
    )
    def test_apply_gpu_finetuned_cpu_safety_vector(self):
        wrapper = self._wrapper()  # safety vector on CPU
        ft = {"w": torch.full((4,), 2.0, device="cuda")}
        out = wrapper.apply(ft, alpha=1.0)  # must not raise
        assert out["w"].is_cuda


# ── Bug 8: unlearn batches must survive multi-epoch re-iteration ──────────────
class TestUnlearnBatchesReiterable:
    def test_device_batches_iterates_more_than_once(self):
        from safetune.runner.unlearn._base import _DeviceBatches

        db = _DeviceBatches([{"x": 1}, {"x": 2}], move=lambda b: b)
        assert [r["x"] for r in db] == [1, 2]
        assert [r["x"] for r in db] == [1, 2]  # a generator would be empty here
        assert len(db) == 2

    def test_to_device_returns_a_reiterable_not_a_generator(self):
        import inspect
        from safetune.runner.unlearn._base import _UnlearnBase

        src = inspect.getsource(_UnlearnBase._to_device)
        assert "_DeviceBatches" in src
        assert "return (_move(b) for b in batches)" not in src


# ── Bug 9: unlearn eval() must forward backend to the safety half too ─────────
class TestUnlearnEvalForwardsBackend:
    def test_backend_in_safety_whitelist(self):
        import inspect
        from safetune.runner.unlearn._base import _UnlearnBase

        src = inspect.getsource(_UnlearnBase.eval)
        safety_seg = src.split("eval_utility")[0]  # the eval_safety call + its kwargs
        assert '"backend"' in safety_seg


# ── Bug 10: harden CLI tokenizer / STAR-DSS safety signal ─────────────────────
class TestHardenCliPadAndSafetyLabel:
    def test_cli_loader_sets_a_pad_token(self):
        import inspect
        from safetune import cli

        assert "pad_token" in inspect.getsource(cli._load_model_and_tok)

    def test_ensure_tokenized_carries_kind_from_is_safe(self, monkeypatch):
        from datasets import Dataset
        import safetune.runner.utils.data_utils as du
        from safetune.cli import _ensure_tokenized

        monkeypatch.setattr(
            du, "_tokenize_qa_rows",
            lambda tok, rows, max_len: [
                {"input_ids": [0], "attention_mask": [1], "labels": [0]} for _ in rows
            ],
        )
        ds = Dataset.from_list([
            {"prompt": "a", "response": "b", "is_safe": True},
            {"prompt": "c", "response": "d", "is_safe": False},
        ])
        out = _ensure_tokenized(ds, tok=object())
        assert out["kind"] == ["benign", "harmful"]


# ── Bug 11: eval backend empty-string env var must still auto-select ──────────
class TestResolveBackendEmptyString:
    def test_empty_string_backend_autoselects(self, monkeypatch):
        from safetune.runner.utils import eval_runner as er

        monkeypatch.setattr(er, "_EVAL_BACKEND", "")
        monkeypatch.setattr(er, "_fast_backend_ready", lambda: False)
        assert er._resolve_backend() == "hf"
        monkeypatch.setattr(er, "_fast_backend_ready", lambda: True)
        assert er._resolve_backend() == "vllm"


# ── Bug 12: eval_utility must reclaim between groups and drop stale files ──────
class TestEvalUtilityRobustness:
    def test_reclaims_between_task_groups(self):
        import inspect
        from safetune.runner.utils import eval_runner as er

        src = inspect.getsource(er.eval_utility)
        # once before the loop + once per group inside it
        assert src.count("reclaim_gpu_memory()") >= 2

    def test_cleans_stale_lmeval_outputs(self):
        import inspect
        from safetune.runner.utils import eval_runner as er

        src = inspect.getsource(er.eval_utility)
        assert "os.remove(stale)" in src


# ── Issue #37: utility_metrics must match its tag exactly, not as a prefix ─────
class TestUtilityMetricsTagExact:
    def _write(self, lmeval_dir, name, val):
        import json
        lmeval_dir.mkdir(parents=True, exist_ok=True)
        (lmeval_dir / name).write_text(
            json.dumps({"results": {"gsm8k": {"exact_match,none": val}}})
        )

    def test_none_tag_does_not_read_code_drift_file(self, tmp_path):
        import os
        from safetune.runner.utils import eval_runner as er

        lmeval = tmp_path / "lmeval"
        folder = "ckpt"
        # This run's merged file (drift_task=None → gsm8k_ifeval_wikitext).
        self._write(lmeval, f"{folder}__base__gsm8k_ifeval_wikitext.json", 0.42)
        # A newer code-drift run sharing the folder — its longer tag starts with
        # ours, so the old `{tag}*` glob would grab it as the newest match.
        code = f"{folder}__base__gsm8k_ifeval_wikitext_humaneval_mbpp.json"
        self._write(lmeval, code, 0.99)
        os.utime(lmeval / code, (10**10, 10**10))  # make it the newest file

        out = er.utility_metrics(folder, drift_task=None, results_dir=str(tmp_path))
        assert out.get("gsm8k") == 0.42  # our file, not the 0.99 code-drift file


# ── Bug 13: summary markdown must tolerate a present-but-None metric ──────────
class TestSummaryMdToleratesNone:
    def test_write_summary_md_with_none_metric(self, tmp_path):
        from safetune.runner.utils.results_writer import ResultsWriter

        w = ResultsWriter("harden", results_dir=str(tmp_path))
        w.append({
            "method": "M", "variant": "v",
            "metrics": {"harmbench_refusal": None, "gsm8k": None},
            "safety_mean": None, "utility_mean": None,
        })
        path = w.write_summary_md()  # must not raise
        assert path and "nan" in Path(path).read_text()


# ── Issue #36: CLI gives a targeted hint for programmatic-only harden methods ─
class TestProgrammaticOnlyHardenHint:
    def _args(self, algo):
        import argparse
        return argparse.Namespace(algo=algo, model="m", output=None)

    def test_programmatic_only_methods_point_to_python_api(self, capsys):
        from safetune import cli

        for algo, cls in cli._PROGRAMMATIC_ONLY_HARDEN.items():
            with pytest.raises(SystemExit):
                cli._do_harden(self._args(algo))
            out = capsys.readouterr().out
            assert "programmatically" in out and cls in out

    def test_truly_unknown_method_still_generic(self, capsys):
        from safetune import cli

        with pytest.raises(SystemExit):
            cli._do_harden(self._args("definitely-not-a-method"))
        assert "Unknown harden method" in capsys.readouterr().out
        
        
# ── Issue #33: eval_safety must not report success when nothing was scored ────
class TestEvalSafetySignalsFailure:
    def test_returns_nonzero_when_no_benches_scored(self, tmp_path, monkeypatch):
        import safetune.evaluate.pipeline as pipe
        from safetune.runner.utils import eval_runner as er

        # Simulate a total generation failure: generate returns {}, nothing is
        # written, and judging/advbench/reclaim are no-ops → no scored files.
        monkeypatch.setattr(pipe, "generate_bench_responses", lambda *a, **k: {})
        monkeypatch.setattr(pipe, "write_bench_jsonl", lambda *a, **k: None)
        monkeypatch.setattr(er, "_score_with_judges", lambda *a, **k: None)
        monkeypatch.setattr(er, "_eval_advbench_inline", lambda *a, **k: None)
        monkeypatch.setattr(er, "reclaim_gpu_memory", lambda *a, **k: None)

        rc, msg = er.eval_safety("ckpt", "some/model", results_dir=str(tmp_path))
        assert rc != 0 and "no scored benchmarks" in msg

    def test_returns_zero_when_a_scored_file_exists(self, tmp_path, monkeypatch):
        import safetune.evaluate.pipeline as pipe
        from safetune.runner.utils import eval_runner as er

        monkeypatch.setattr(pipe, "generate_bench_responses", lambda *a, **k: {})
        monkeypatch.setattr(pipe, "write_bench_jsonl", lambda *a, **k: None)
        monkeypatch.setattr(er, "_score_with_judges", lambda *a, **k: None)
        monkeypatch.setattr(er, "_eval_advbench_inline", lambda *a, **k: None)
        monkeypatch.setattr(er, "reclaim_gpu_memory", lambda *a, **k: None)
        # a non-empty scored file present → success
        safety = tmp_path / "safety"
        safety.mkdir(parents=True, exist_ok=True)
        (safety / "ckpt__base__harmbench_scored.jsonl").write_text('{"x":1}\n')

        rc, _ = er.eval_safety("ckpt", "some/model", results_dir=str(tmp_path))
        assert rc == 0
        
        
# ── Issue #35: SAP must feed a real contrastive safety batch, not all-masked ──
@pytest.mark.skipif(torch is None, reason="torch required")
class TestSapContrastiveSignal:
    def test_contrastive_collator_keeps_distinct_rejected_labels(self):
        from safetune.runner.harden._base import _sap_contrastive_collator

        feats = [{"input_ids": [1, 2], "attention_mask": [1, 1],
                  "chosen_labels": [1, 2], "rejected_labels": [3, 4]}]
        b = _sap_contrastive_collator(feats)
        assert set(b) == {"input_ids", "attention_mask", "chosen_labels", "rejected_labels"}
        # the old _sap_safety_collator forced rejected_labels to all -100
        assert not bool((b["rejected_labels"] == -100).all())
        assert not bool(torch.equal(b["chosen_labels"], b["rejected_labels"]))

    def test_sap_runner_defaults_to_contrastive_dataset(self):
        import inspect
        from safetune.runner.harden._regularization import SAPTrainer

        src = inspect.getsource(SAPTrainer.train)
        assert "sap_contrastive_dataset" in src
        assert "_sap_contrastive_collator" in src


# ── Bug 4: eval demo uses a valid abliteration mode ──────────────────────────
class TestExampleAbliterationMode:
    def _examples_dir(self):
        import safetune

        return Path(safetune.__file__).resolve().parents[2] / "examples"

    def test_eval_demos_use_a_valid_mode(self):
        demos = self._examples_dir() / "demos"
        if not demos.exists():
            pytest.skip("examples not packaged in this install")
        for name in ("eval_model.py", "evaluate_demo.py"):
            text = (demos / name).read_text()
            assert 'mode="ablate"' not in text and "mode='ablate'" not in text, name
            assert "runtime_ablate" in text, name
