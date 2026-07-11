"""Base class and helpers for the unlearn runner."""
from __future__ import annotations
import os

from typing import Optional

from safetune.runner.utils.eval_runner import eval_safety, eval_utility, all_metrics
from safetune.runner.utils.results_writer import ResultsWriter, DEFAULT_RESULTS_DIR
from safetune.runner.utils.model_utils import lora_wrap, free


U = None
_unlearn_imports_done = False


def _ensure_unlearn_imports():
    global U, _unlearn_imports_done
    if not _unlearn_imports_done:
        import safetune.unlearn as _U
        U = _U
        _unlearn_imports_done = True


_LORA_METHODS = {"NPO", "FLAT", "SimDPO"}


class _DeviceBatches:
    """Re-iterable view over a batch list that moves each batch to the device
    on demand. Unlike a generator, it can be iterated once per epoch."""

    def __init__(self, batches, move):
        self._batches = batches
        self._move = move

    def __iter__(self):
        return (self._move(b) for b in self._batches)

    def __len__(self):
        return len(self._batches)


def _derive_model_id(model_id, model=None, tokenizer=None) -> str:
    if model_id is not None:
        return str(model_id)
    cand = getattr(tokenizer, "name_or_path", None)
    if not cand:
        cand = getattr(getattr(model, "config", None), "_name_or_path", None)
    return str(cand) if cand else "model"


class _UnlearnBase:
    PILLAR = "unlearn"
    METHOD: str = ""
    USE_LORA: bool = False

    def __init__(
        self,
        model=None,
        *,
        model_id=None,
        results_dir: str = None,
        drift_task: str = None,
        **kwargs,
    ):
        self.model_id = _derive_model_id(model_id, model)
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

    def _wrap_lora(self, model):
        if self.USE_LORA:
            return lora_wrap(model)
        return model

    def _maybe_merge(self, model):
        if hasattr(model, "merge_and_unload"):
            return model.merge_and_unload()
        return model

    def _to_device(self, batches):
        import torch
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            return batches

        def _move_value(v):
            if isinstance(v, torch.Tensor):
                t = v.unsqueeze(0) if v.dim() == 1 else v
                return t.to(device)
            if isinstance(v, dict):
                return {sk: _move_value(sv) for sk, sv in v.items()}
            if isinstance(v, list):
                return torch.tensor([v]).to(device)
            return v

        def _move(batch):
            return {k: _move_value(v) for k, v in batch.items()}

        # Multi-epoch unlearn trainers iterate their batch stream once per epoch
        # (`for epoch in range(...)`: `for f in forget_batches` / `zip(...)`). A
        # one-shot generator would be exhausted after epoch 1, silently turning
        # epochs 2+ into no-ops (~1/5 of the work at the default 5 epochs).
        # Return a re-iterable view that moves each batch to the device lazily on
        # every pass, so only one batch is resident on the GPU at a time.
        return _DeviceBatches(list(batches), _move)

    def unlearn(self, forget, retain, **kwargs):
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
        import torch
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
                        if k in ("gpu", "backend", "base_model_name")})
        return all_metrics(folder_name, drift_task=dt, results_dir=self.results_dir)

    def save_checkpoint(self, unlearned_model, tokenizer=None, name: str = "unlearn") -> str:
        from safetune.runner.utils.model_utils import save_checkpoint, load_tok
        tok = tokenizer or load_tok(self.model_id)
        ckpt_dir = os.path.join(self.results_dir, "checkpoints")
        return save_checkpoint(unlearned_model, tok, name, out_dir=ckpt_dir)

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
            name = self.METHOD.lower().replace(" ", "_") or "unlearn"
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
