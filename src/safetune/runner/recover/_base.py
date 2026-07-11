"""Recover runner — base class and helpers."""
from __future__ import annotations
import os

from typing import Optional

import torch

from safetune.runner.utils.eval_runner import eval_safety, eval_utility, all_metrics
from safetune.runner.utils.results_writer import ResultsWriter, DEFAULT_RESULTS_DIR
from safetune.runner.utils.model_utils import free, derive_model_id




class _RecoverBase:
    PILLAR = "recover"
    METHOD: str = ""

    def __init__(
        self,
        model=None,
        *,
        model_id=None,
        results_dir: str = None,
        drift_task: str = None,
        **kwargs,
    ):
        self.model_id = derive_model_id(model_id, model)
        self.model = model
        self.results_dir = results_dir or DEFAULT_RESULTS_DIR
        self.drift_task = drift_task
        self._extra = kwargs

    @property
    def model(self):
        if getattr(self, '_model', None) is None:
            from safetune.runner.utils.model_utils import load_model
            self._model = load_model(self.model_id)
        return self._model

    @model.setter
    def model(self, v):
        self._model = v

    def apply(self, **kwargs):
        """Apply the post-hoc safety patch and return the patched model.

        Subclasses must override this method. The contract:
          - Mutate ``self.model`` in-place and/or return a new model object.
          - Never write to disk here; the caller calls ``save_checkpoint`` separately.
          - Accept any method-specific overrides as keyword args.
        """
        raise NotImplementedError

    def eval(
        self,
        folder_name: str,
        model_path: str,
        *,
        drift_task: str = None,
        **kwargs,
    ) -> dict:
        import gc
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        dt = drift_task or self.drift_task
        gpu_mem = kwargs.pop(
            "gpu_memory_utilization",
            float(os.environ.get("SAFETUNE_GPU_MEM", "0.75")),
        )
        eval_safety(folder_name, model_path, results_dir=self.results_dir,
                    gpu_memory_utilization=gpu_mem,
                    **{k: v for k, v in kwargs.items()
                       if k in ("gpu", "backend", "base_model_name")})
        eval_utility(folder_name, model_path, drift_task=dt,
                     results_dir=self.results_dir,
                     gpu_mem=gpu_mem,
                     **{k: v for k, v in kwargs.items()
                        if k in ("gpu", "backend", "base_model_name", "limit")})
        return all_metrics(folder_name, drift_task=dt, results_dir=self.results_dir)

    def save_checkpoint(self, patched_model, tokenizer=None, name: str = "recover") -> str:
        from safetune.runner.utils.model_utils import save_checkpoint, load_tok
        tok = tokenizer or load_tok(self.model_id)
        ckpt_dir = os.path.join(self.results_dir, "checkpoints")
        return save_checkpoint(patched_model, tok, name, out_dir=ckpt_dir)

    def save_results(
        self,
        metrics: dict,
        *,
        variant: str = "default",
    ) -> None:
        from safetune.runner.utils.eval_runner import safety_mean, utility_mean
        record = {
            "method": self.METHOD,
            "variant": variant,
            "metrics": metrics,
            "safety_mean": safety_mean(metrics),
            "utility_mean": utility_mean(metrics, drift_task=self.drift_task),
        }
        ResultsWriter(self.PILLAR, results_dir=self.results_dir).append(record)

    def evaluate(self, model_or_path, domain: str = None) -> dict:
        from safetune.runner.utils.eval_runner import safety_mean, utility_mean
        from safetune.runner.utils.model_utils import load_tok
        if not isinstance(model_or_path, str):
            tok = load_tok(self.model_id)
            name = self.METHOD.lower().replace(" ", "_") or "recover"
            path = os.path.join(self.results_dir, "checkpoints", name)
            os.makedirs(path, exist_ok=True)
            model_or_path.save_pretrained(path)
            tok.save_pretrained(path)
            model_or_path = path
        m = self.eval(os.path.basename(model_or_path), model_or_path, drift_task=domain)
        print(m)
        from safetune.runner.utils.eval_runner import fmt_metric
        print(f"[eval] {os.path.basename(model_or_path)}  safety={fmt_metric(safety_mean(m))}  utility={fmt_metric(utility_mean(m, drift_task=domain))}")
        return m
