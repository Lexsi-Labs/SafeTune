"""Build a vLLM generation backend from any calibrated SafeTune steer model.

Usage
-----
>>> from safetune.steer.backends.vllm_eval import build_vllm_eval_backend
>>> backend = build_vllm_eval_backend(model_id, wrapped_model)
>>> responses = backend.generate(prompts, max_tokens=256)

The returned backend exposes:
    .generate(prompts, *, max_tokens, temperature, apply_chat_template) -> List[str]

Routing table
-------------
+-----------------------------+---------------------------+--------------------+
| SafeTune class              | Backend                   | Fidelity           |
+=============================+===========================+====================+
| CAAModel                    | VLLMHookSteer             | Full               |
| STAModel                    | VLLMHookSteer             | Full               |
| SafeSteerModel              | VLLMHookSteer             | Full               |
| RefusalDirectionModel       | VLLMHookSteer             | Full               |
| SCANSModel                  | VLLMHookSteer             | Partial (sign)     |
| AlphaSteerModel             | VLLMHookSteer             | Partial (prefill)  |
| CASTModel                   | VLLMHookSteer             | Approx (no gate)   |
| CircuitBreakerRRModel       | VLLMHookSteer             | Approx (add, not   |
|                             |                           | reroute)           |
| SafeDecodingProcessor       | VLLMDecodeSteer           | Full               |
| ContrastiveDecodingProcessor| VLLMDecodeSteer           | Full               |
| ProxyTuningProcessor        | VLLMDecodeSteer           | Full               |
| NudgingProcessor            | VLLMDecodeSteer           | Full               |
| CircuitBreakerModel         | _VLLMPlainBackend         | Full (passthrough) |
| RepBendModel                | _VLLMPlainBackend         | Full (passthrough) |
| TARModel                    | _VLLMPlainBackend         | Full (passthrough) |
| RRFAEnsemble                | _VLLMPlainBackend         | Full (passthrough) |
+-----------------------------+---------------------------+--------------------+

Not supported (raises NotImplementedError)
------------------------------------------
* AdaSteerModel     — adaptive per-token coefficient (logistic on hidden state)
                      requires live logistic eval in the worker subprocess
* SafeSwitchModel   — two-stage prober MLP gate + LM head swap, not wired into
                      the hook worker
* LinearProbeGuardModel — probe-based routing (return canned refusal or pass);
                          no vector-space representation for vLLM hook
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# ── Classification tables ──────────────────────────────────────────────────────

_HOOK_METHODS = frozenset({
    "CAAModel", "STAModel", "SafeSteerModel", "RefusalDirectionModel",
    "SCANSModel", "AlphaSteerModel", "CASTModel", "CircuitBreakerRRModel",
})

_DECODE_METHODS = frozenset({
    "SafeDecodingProcessor", "ContrastiveDecodingProcessor",
    "ProxyTuningProcessor", "NudgingProcessor",
})

_PLAIN_METHODS = frozenset({
    "CircuitBreakerModel", "RepBendModel", "TARModel", "RRFAEnsemble",
})

VLLM_UNSUPPORTED: Dict[str, str] = {
    "AdaSteerModel": (
        "requires live adaptive coefficient (logistic regression on hidden state) "
        "inside the worker — not yet wired into MultiLayerSteerWorker"
    ),
    "SafeSwitchModel": (
        "requires a two-stage prober MLP gate + LM head swap inside the worker "
        "— not implemented"
    ),
    "LinearProbeGuardModel": (
        "inference-time routing (probe gate → canned refusal or passthrough); "
        "no vector-space representation exists for vLLM hook"
    ),
}


# ── Plain vLLM backend (weight-editing / passthrough methods) ─────────────────

class _VLLMPlainBackend:
    """Wraps a plain vLLM engine — no steering — for weight-space methods.

    Weight-editing methods (CircuitBreaker, RepBend, TAR) bake their
    intervention into model weights at training time and are passthrough at
    inference.  Evaluation via vanilla vLLM is exact.
    """

    def __init__(self, model_id: str, **vllm_kwargs: Any) -> None:
        from vllm import LLM
        kw = dict(dtype="bfloat16", enforce_eager=True,
                  gpu_memory_utilization=0.85, max_model_len=4096)
        kw.update(vllm_kwargs)
        self.llm = LLM(model=model_id, **kw)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        apply_chat_template: bool = True,
        **_: Any,
    ) -> List[str]:
        from vllm import SamplingParams
        rendered = _render_prompts(self.tokenizer, list(prompts), apply_chat_template)
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        return [o.outputs[0].text for o in self.llm.generate(rendered, sp)]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _render_prompts(
    tokenizer: Any,
    prompts: List[str],
    apply_chat_template: bool,
) -> List[str]:
    if not apply_chat_template:
        return prompts
    has_template = getattr(tokenizer, "chat_template", None) is not None
    if not has_template:
        return prompts
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]


def _guide_model_id(processor: Any, fallback: str) -> str:
    """Try to extract the HF model id from a GuidedLogitsProcessor's guide."""
    guide = getattr(processor, "guide", None)
    if guide is not None:
        cfg = getattr(guide, "config", None)
        name_or_path = getattr(cfg, "_name_or_path", None)
        if name_or_path:
            return name_or_path
    return fallback


# ── Spec extraction helpers ───────────────────────────────────────────────────

def _extract_cast_spec(cast_model: Any):
    """Build a SteerSpec from CASTModel (unconditional approximation).

    CASTModel's gate (conditional on hidden state) is dropped; the vectors
    are applied unconditionally at fixed alpha.  This is an approximation.
    """
    from safetune.steer.backends.vllm_hook import SteerSpec
    vectors = {int(k): v.detach()
               for k, v in cast_model.steering_vectors.items()}
    return SteerSpec(op="add", vectors=vectors,
                     coeff=float(cast_model.alpha), method="cast_approx")


def _extract_cb_rr_spec(cb_rr_model: Any):
    """Build a SteerSpec from CircuitBreakerRRModel (additive approximation).

    The true circuit-breaker reroute operation re-routes activations through
    refusal directions rather than additively steering.  This spec applies an
    additive intervention with the same direction vectors, which is an
    approximation useful for evaluation comparisons.
    """
    from safetune.steer.backends.vllm_hook import SteerSpec
    vectors = {int(k): v.detach() for k, v in cb_rr_model.directions.items()}
    coeff = float(getattr(getattr(cb_rr_model, "config", None), "strength", 1.0))
    return SteerSpec(op="add", vectors=vectors, coeff=coeff, method="cb_rr_approx")


def _extract_decode_steer_spec(processor: Any, model_id: str):
    """Build a DecodeSteerSpec from a calibrated GuidedLogitsProcessor."""
    from safetune.steer.backends.vllm_logits import DecodeSteerSpec

    cls = type(processor).__name__
    guide_id = _guide_model_id(processor, model_id)

    if cls == "SafeDecodingProcessor":
        cfg = processor.config
        return DecodeSteerSpec(
            method="safedecoding",
            aux_model=guide_id,
            params={
                "alpha": float(cfg.alpha),
                "first_m": int(cfg.window()),
                "num_common_tokens": int(cfg.num_common_tokens),
                "top_k": int(cfg.top_k),
            },
        )

    if cls == "ContrastiveDecodingProcessor":
        cfg = processor.config
        return DecodeSteerSpec(
            method="contrastive",
            aux_model=guide_id,
            params={"alpha": float(cfg.alpha)},
        )

    if cls == "ProxyTuningProcessor":
        cfg = processor.config
        # ProxyTuning: aux_model=expert(proxy-tuned guide), aux_model_2=antiexpert(base)
        proxy_base = getattr(processor, "proxy_base", None)
        antiexpert_id = model_id
        if proxy_base is not None:
            pb_cfg = getattr(proxy_base, "config", None)
            antiexpert_id = getattr(pb_cfg, "_name_or_path", None) or model_id
        return DecodeSteerSpec(
            method="proxy_tuning",
            aux_model=guide_id,
            aux_model_2=antiexpert_id,
            params={"scale": float(cfg.scale)},
        )

    if cls == "NudgingProcessor":
        cfg = processor.config
        return DecodeSteerSpec(
            method="nudging",
            aux_model=guide_id,
            params={"top_prob_thres": float(cfg.top_prob_thres)},
        )

    raise NotImplementedError(
        f"_extract_decode_steer_spec: unsupported processor class {cls!r}"
    )


# ── Main dispatch ─────────────────────────────────────────────────────────────

def build_vllm_eval_backend(
    model_id: str,
    wrapped_model: Any,
    **vllm_kwargs: Any,
) -> Any:
    """Build a vLLM generation backend for a calibrated SafeTune steer model.

    Returns an object with a ``.generate(prompts, *, max_tokens, temperature,
    apply_chat_template) -> List[str]`` interface.

    Args:
        model_id: HF Hub id or local path of the *base* model (before wrapping).
        wrapped_model: A calibrated SafeTune steer-model wrapper (e.g. CAAModel,
            SafeDecodingProcessor, CircuitBreakerModel, …).
        **vllm_kwargs: Forwarded to the vLLM engine constructor
            (``gpu_memory_utilization``, ``max_model_len``, …).

    Raises:
        NotImplementedError: For methods that cannot be served via vLLM
            (AdaSteerModel, SafeSwitchModel, LinearProbeGuardModel).
    """
    from safetune.steer.backends.vllm_hook import extract_steer_spec, VLLMHookSteer
    from safetune.steer.backends.vllm_logits import VLLMDecodeSteer

    cls = type(wrapped_model).__name__

    if cls in VLLM_UNSUPPORTED:
        raise NotImplementedError(
            f"{cls} cannot be evaluated via vLLM: {VLLM_UNSUPPORTED[cls]}"
        )

    # Weight-editing passthrough: exact fidelity via plain vLLM.
    if cls in _PLAIN_METHODS:
        return _VLLMPlainBackend(model_id, **vllm_kwargs)

    # Logits-processor decoding methods.
    if cls in _DECODE_METHODS:
        spec = _extract_decode_steer_spec(wrapped_model, model_id)
        kw = dict(gpu_memory_utilization=0.55)
        kw.update(vllm_kwargs)
        return VLLMDecodeSteer(target_model=model_id, spec=spec, **kw)

    # Activation-steering: CAST and CircuitBreakerRR need custom spec extraction.
    if cls == "CASTModel":
        spec = _extract_cast_spec(wrapped_model)
        return VLLMHookSteer(model=model_id, spec=spec, **vllm_kwargs)

    if cls == "CircuitBreakerRRModel":
        spec = _extract_cb_rr_spec(wrapped_model)
        return VLLMHookSteer(model=model_id, spec=spec, **vllm_kwargs)

    # General hook-based methods: CAAModel, STAModel, SCANSModel, etc.
    # extract_steer_spec raises NotImplementedError for AdaSteer / SafeSwitch.
    spec = extract_steer_spec(wrapped_model)
    return VLLMHookSteer(model=model_id, spec=spec, **vllm_kwargs)


__all__ = [
    "VLLM_UNSUPPORTED",
    "build_vllm_eval_backend",
]
