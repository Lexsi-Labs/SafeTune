"""Regression tests for the CLI / runner fixes from the doc-change audit.

Each test pins a bug that shipped on the ``doc-change`` branch:

1. ``safetune unlearn`` crashed because the tokenizer was passed as an illegal
   positional arg to keyword-only unlearn trainer ``__init__``s.
2. DeRTa hardening silently disabled its harmful/safe signal because
   ``set_format`` dropped the ``safe`` column.
3. ``RECOVER_REGISTRY`` omitted three shipped methods (pke/safereact/scrub).
4. ``--train-split`` defaulted to a BeaverTails-only split for every dataset.
5. ``SafeTuneConfig.method_kwargs`` were never forwarded to the trainer.

These are deliberately model-free / device-free so they run on CPU, CUDA and
Apple-Silicon (MPS) alike.
"""
import argparse
import inspect

import pytest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ── Fix 1: unlearn CLI no longer passes tokenizer positionally ────────────────

@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestUnlearnCliConstruction:
    def test_do_unlearn_does_not_pass_tokenizer_positionally(self):
        from safetune.cli import _do_unlearn
        src = inspect.getsource(_do_unlearn)
        # The blocker was `TrainerClass(model, tok, ...)`.
        assert "TrainerClass(model, tok" not in src

    @pytest.mark.parametrize("alias", [
        "rmu", "npo", "ga", "graddiff", "flat", "simdpo",
    ])
    def test_every_unlearn_trainer_constructs_the_cli_way(self, alias):
        """Reproduce the exact CLI construction; must not raise TypeError."""
        from safetune.runner._registry import UNLEARN_REGISTRY
        import safetune.runner.unlearn as U
        from safetune.cli import _trainer_kwargs

        TrainerClass = getattr(U, UNLEARN_REGISTRY[alias])
        ns = argparse.Namespace(epochs=1, batch_size=1, lr=5e-5,
                                precision="bf16", method_kwargs={})
        # model=None is fine: the trainer lazy-loads from model_id only if used.
        trainer = TrainerClass(None, model_id="dummy/model", **_trainer_kwargs(ns))
        assert trainer is not None


# ── Fix 2: DeRTa keeps the `safe` column ──────────────────────────────────────

@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestDeRTaSafeColumn:
    def test_collator_propagates_real_safe_values(self):
        from datasets import Dataset
        from safetune.runner.harden._base import _derta_collator

        rows = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1],
             "labels": [1, 2, 3], "safe": True},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1],
             "labels": [4, 5, 6], "safe": False},
        ]
        ds = Dataset.from_list(rows)
        # Mirror the production formatting (must include "safe").
        ds.set_format(type="torch",
                      columns=["input_ids", "attention_mask", "labels", "safe"])
        batch = _derta_collator([ds[0], ds[1]])
        # The bug produced [True, True]; the fix preserves [True, False].
        assert batch["safe"].tolist() == [True, False]

    def test_production_set_format_includes_safe(self):
        """Guard the fix itself: DeRTaTrainer.train must keep `safe` formatted."""
        from safetune.runner.harden._data_shaping import DeRTaTrainer
        src = inspect.getsource(DeRTaTrainer.train)
        # find the derta set_format line and assert it lists "safe"
        fmt_lines = [l for l in src.splitlines() if "set_format" in l]
        assert fmt_lines, "DeRTaTrainer.train no longer calls set_format"
        assert any('"safe"' in l or "'safe'" in l for l in fmt_lines), (
            "DeRTa set_format dropped the 'safe' column — harmful/safe signal "
            "will be silently disabled"
        )


# ── Fix 3: registry exposes every shipped recover method ──────────────────────

class TestRecoverRegistryCompleteness:
    @pytest.mark.parametrize("alias,cls", [
        ("pke", "PKETrainer"),
        ("safereact", "SafeReActTrainer"),
        ("scrub", "SCRUBTrainer"),
    ])
    def test_missing_methods_now_registered(self, alias, cls):
        from safetune.runner._registry import RECOVER_REGISTRY
        assert RECOVER_REGISTRY.get(alias) == cls

    @pytest.mark.parametrize("pillar", ["harden", "recover", "unlearn"])
    def test_every_registry_entry_resolves_to_an_importable_class(self, pillar):
        import importlib
        from safetune.runner import _registry
        registry = getattr(_registry, f"{pillar.upper()}_REGISTRY")
        mod = importlib.import_module(f"safetune.runner.{pillar}")
        unresolved = [cls for cls in registry.values() if not hasattr(mod, cls)]
        assert not unresolved, f"{pillar}: unresolved registry classes {unresolved}"


# ── Device-agnostic: harden runners must not hardcode CUDA ────────────────────

class TestNoHardcodedCuda:
    """DOOR / gradient-surgery / regularization runners loaded their reference
    model with device_map="cuda" (and moved tensors with .to("cuda")), which
    crashed on Apple-Silicon / CPU. Guard against re-introducing that."""

    @pytest.mark.parametrize("module", [
        "safetune.runner.harden._tamper_resistant",
        "safetune.runner.harden._gradient_surgery",
        "safetune.runner.harden._regularization",
    ])
    def test_no_hardcoded_cuda_device(self, module):
        import importlib, inspect
        src = inspect.getsource(importlib.import_module(module))
        assert 'device_map="cuda"' not in src, f"{module} hardcodes device_map=cuda"
        assert '.to("cuda")' not in src, f"{module} hardcodes .to('cuda')"


# ── AAQ forwards simulate_quantization (was stored but silently dropped) ──────

@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestAAQForwardsSimulateQuantization:
    def test_simulate_quantization_reaches_apply_aaq(self, monkeypatch):
        import torch.nn as nn
        import safetune.recover as R
        from safetune.runner.recover import AAQTrainer
        captured = {}
        monkeypatch.setattr(R, "apply_aaq",
                            lambda model, **kw: captured.update(kw) or model)
        stub_model = nn.Linear(2, 2)  # real module so self.model doesn't lazy-load
        # trainer default is True; must not silently become apply_aaq's default (False)
        AAQTrainer(stub_model, simulate_quantization=True).apply()
        assert captured.get("simulate_quantization") is True
        captured.clear()
        AAQTrainer(stub_model, simulate_quantization=False).apply()
        assert captured.get("simulate_quantization") is False


# ── Fix 4: dataset-aware --train-split default ────────────────────────────────

@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestTrainSplitDefault:
    def _run(self, monkeypatch, train_dataset, train_split):
        captured = {}

        def fake_load_dataset(name, split=None, **kw):
            captured["source"] = "hf"
            captured["name"] = name
            captured["split"] = split
            return "DS"

        def fake_load_beavertails(split=None, **kw):
            captured["source"] = "beavertails"
            captured["split"] = split
            return "DS"

        import datasets
        monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
        import safetune.data as sd
        monkeypatch.setattr(sd, "load_beavertails", fake_load_beavertails,
                            raising=False)

        from safetune.cli import _load_train_dataset
        ns = argparse.Namespace(train_dataset=train_dataset,
                                train_split=train_split)
        _load_train_dataset(ns)
        return captured

    def test_beavertails_defaults_to_30k_train(self, monkeypatch):
        c = self._run(monkeypatch, "beavertails", None)
        assert c["source"] == "beavertails" and c["split"] == "30k_train"

    def test_other_dataset_defaults_to_train_not_30k(self, monkeypatch):
        c = self._run(monkeypatch, "tatsu-lab/alpaca", None)
        assert c["source"] == "hf" and c["split"] == "train"

    def test_explicit_split_is_respected(self, monkeypatch):
        c = self._run(monkeypatch, "tatsu-lab/alpaca", "validation")
        assert c["split"] == "validation"


# ── Fix 5: method_kwargs are forwarded to the trainer ─────────────────────────

class TestMethodKwargsForwarding:
    def test_trainer_kwargs_merges_method_kwargs(self):
        from safetune.cli import _trainer_kwargs
        ns = argparse.Namespace(epochs=2, batch_size=4, lr=1e-5,
                                precision="bf16",
                                method_kwargs={"lisa_rho": 0.5, "rank": 8})
        kw = _trainer_kwargs(ns)
        assert kw["lisa_rho"] == 0.5 and kw["rank"] == 8
        # standard kwargs still present
        assert kw["epochs"] == 2 and kw["bf16"] is True

    def test_trainer_kwargs_without_method_kwargs_is_safe(self):
        from safetune.cli import _trainer_kwargs
        ns = argparse.Namespace(epochs=1, batch_size=1, lr=5e-5,
                                precision="fp32")
        kw = _trainer_kwargs(ns)  # no method_kwargs attribute at all
        assert kw["bf16"] is False

    def test_config_unknown_keys_land_in_method_kwargs(self, tmp_path):
        from safetune.config import SafeTuneConfig
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text(
            "command: train\nalgo: lisa\nmodel: dummy\n"
            "epochs: 3\nlisa_rho: 0.2\n"
        )
        cfg = SafeTuneConfig.from_yaml(str(cfg_file))
        assert cfg.method_kwargs == {"lisa_rho": 0.2}
        assert cfg.epochs == 3

    def test_yaml_numeric_fields_are_coerced(self, tmp_path):
        """PyYAML parses `lr: 2e-5` as a str; from_yaml must coerce numeric fields."""
        from safetune.config import SafeTuneConfig
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text(
            "command: train\nmodel: dummy\n"
            "lr: 2e-5\nepochs: 3\nbatch_size: 4\n"   # lr would be a str without coercion
        )
        cfg = SafeTuneConfig.from_yaml(str(cfg_file))
        assert isinstance(cfg.lr, float) and cfg.lr == 2e-5
        assert isinstance(cfg.epochs, int) and cfg.epochs == 3
        assert isinstance(cfg.batch_size, int) and cfg.batch_size == 4

    def test_parse_args_attaches_method_kwargs(self, tmp_path, monkeypatch):
        import sys
        from safetune.cli import parse_args
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text("command: train\nmodel: dummy\nlisa_rho: 0.9\n")
        monkeypatch.setattr(sys, "argv",
                            ["safetune", "train", "--config", str(cfg_file)])
        args = parse_args()
        assert getattr(args, "method_kwargs", None) == {"lisa_rho": 0.9}
