from __future__ import annotations
import os

from typing import Optional

import torch

from safetune.runner.utils.eval_runner import (
    eval_safety, eval_utility, all_metrics, safety_mean, utility_mean, utility_metrics,
)
from safetune.runner.utils.results_writer import ResultsWriter, DEFAULT_RESULTS_DIR
from safetune.runner.utils.model_utils import free


S = None
_steer_imports_done = False


def _ensure_steer_imports():
    global S, _steer_imports_done
    if not _steer_imports_done:
        import safetune.steer as _S
        S = _S
        _steer_imports_done = True


def _derive_model_id(model_id, model=None, tokenizer=None) -> str:
    if model_id is not None:
        return str(model_id)
    cand = getattr(tokenizer, "name_or_path", None)
    if not cand:
        cand = getattr(getattr(model, "config", None), "_name_or_path", None)
    return str(cand) if cand else "model"


class _SteerBase:
    PILLAR = "steer"
    METHOD: str = ""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        *,
        model_id=None,
        results_dir: str = None,
        drift_task: str = None,
        **kwargs,
    ):
        self.model_id = _derive_model_id(model_id, model, tokenizer)
        self.model = model
        self.tok = tokenizer
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

    @property
    def tok(self):
        if getattr(self, '_tok', None) is None:
            from safetune.runner.utils.model_utils import load_tok
            self._tok = load_tok(self.model_id)
        return self._tok

    @tok.setter
    def tok(self, v):
        self._tok = v

    def calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  out_dir: str = None, **kwargs):
        _ensure_steer_imports()
        wrapped = self._do_calibrate(harmful, harmless, calib_n=calib_n, **kwargs)
        return wrapped, (out_dir or self.model_id)

    def _do_calibrate(self, *args, **kwargs):
        raise NotImplementedError

    def eval_live(
        self,
        folder_name: str,
        wrapped_model,
        *,
        bench_prompts: dict = None,
        drift_task: str = None,
        **kwargs,
    ) -> dict:
        from safetune.evaluate.suite.evaluate import evaluate
        dt = drift_task or self.drift_task
        from safetune.runner.utils.data_utils import SAFETY_BENCHES
        result = evaluate(
            model=wrapped_model,
            tokenizer=self.tok,
            benchmarks=SAFETY_BENCHES,
            judge="wildguard",
            batch_size=kwargs.get("batch_size", 4),
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            drift_task=dt,
        )
        m: dict = {}
        for bench, bench_result in result.items():
            rr = bench_result.get("refusal_rate")
            if rr is not None:
                key = bench.replace("_v1", "")
                m[f"{key}_refusal"] = rr
        # Runtime steering doesn't change weights, so capability is that of the
        # base model — run the utility eval on model_id (mirrors eval_vllm).
        # Without this call utility_metrics reads nothing and utility_mean is
        # None ("N/A"), even though safety was measured on the steered model.
        eval_utility(folder_name, self.model_id, drift_task=dt,
                     results_dir=self.results_dir)
        m.update(utility_metrics(folder_name, drift_task=dt, results_dir=self.results_dir))
        return m

    def save_results(
        self,
        metrics: dict,
        *,
        variant: str = "default",
    ) -> None:
        record = {
            "method": self.METHOD,
            "variant": variant,
            "metrics": metrics,
            "safety_mean": safety_mean(metrics),
            "utility_mean": utility_mean(metrics, drift_task=self.drift_task),
        }
        ResultsWriter(self.PILLAR, results_dir=self.results_dir).append(record)

    def eval_vllm(
        self,
        wrapped_model,
        *,
        domain: str = None,
        **vllm_kwargs,
    ) -> dict:
        from safetune.steer.backends.vllm_eval import (
            build_vllm_eval_backend, VLLM_UNSUPPORTED,
        )
        from safetune.evaluate.suite.evaluate import evaluate_with_vllm_backend
        from safetune.runner.utils.data_utils import SAFETY_BENCHES

        cls = type(wrapped_model).__name__
        if cls in VLLM_UNSUPPORTED:
            import warnings
            warnings.warn(
                f"{cls} does not support vLLM eval "
                f"({VLLM_UNSUPPORTED[cls]}). Falling back to HF.",
                stacklevel=2,
            )
            name = self.METHOD.lower().replace(" ", "_") or "steer"
            return self.eval_live(name, wrapped_model, drift_task=domain)

        max_new_tokens = vllm_kwargs.pop("max_new_tokens", 256)
        backend = build_vllm_eval_backend(self.model_id, wrapped_model, **vllm_kwargs)

        result = evaluate_with_vllm_backend(
            backend,
            benchmarks=SAFETY_BENCHES,
            judge="wildguard",
            max_new_tokens=max_new_tokens,
            drift_task=domain or self.drift_task,
        )

        m: dict = {}
        for bench, bench_result in result.items():
            rr = bench_result.get("refusal_rate")
            if rr is not None:
                key = bench.replace("_v1", "")
                m[f"{key}_refusal"] = rr

        del backend
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        name = self.METHOD.lower().replace(" ", "_") or "steer"
        dt = domain or self.drift_task
        eval_utility(name, self.model_id, drift_task=dt, results_dir=self.results_dir)
        m.update(utility_metrics(name, drift_task=dt, results_dir=self.results_dir))
        return m

    def evaluate(self, model_or_path, domain: str = None, backend: str = "hf") -> dict:
        if isinstance(model_or_path, str):
            name = os.path.basename(model_or_path)
            eval_safety(name, model_or_path, results_dir=self.results_dir, backend=backend)
            eval_utility(name, model_or_path, drift_task=domain,
                         results_dir=self.results_dir, backend=backend)
            m = all_metrics(name, drift_task=domain, results_dir=self.results_dir)
        elif backend == "vllm":
            m = self.eval_vllm(model_or_path, domain=domain)
        else:
            name = self.METHOD.lower().replace(" ", "_") or "steer"
            m = self.eval_live(name, model_or_path, drift_task=domain)
        print(m)
        s = safety_mean(m)
        u = utility_mean(m, drift_task=domain)
        s_str = f"{s:.3f}" if s is not None else "N/A"
        u_str = f"{u:.3f}" if u is not None else "N/A"
        print(f"[eval] safety={s_str}  utility={u_str}")
        return m

    def _default_calib(self, n=256):
        from safetune.runner.utils.data_utils import refusal_prompt_pairs_large
        return refusal_prompt_pairs_large(n)
