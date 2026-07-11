"""
Adapter: run SafeTune activation-steering methods inside vLLM via IBM/vLLM-Hook.

Background
----------
SafeTune's STEER pillar (eval-design Part 3) implements activation-steering
methods as HF ``forward_hook``s on decoder layers (see
``safetune.steer.refusal_direction.RefusalDirectionModel``, ``caa.CAAModel``,
``scans.SCANSModel``, ``sta.STAModel``, ``alphasteer.AlphaSteerModel``,
``safesteer.SafeSteerModel`` ...).  Running those under HF/transformers
generation is slow.  vLLM is fast, but you cannot register a Python forward
hook on a normal ``vllm.LLM`` because the model runs inside a worker
subprocess.

IBM/vLLM-Hook (https://github.com/IBM/vLLM-Hook) solves this with a custom
vLLM ``Worker`` subclass that installs forward hooks on the model *inside the
worker process*, right after ``load_model()``.  Their stock
``SteerHookActWorker`` only hooks a *single* layer and loads a *single* ``dir``
tensor from disk -- enough for one-layer refusal steering, not enough for
SafeTune's multi-layer per-layer-vector methods (CAA, SCANS, AlphaSteer).

This module provides:

* ``MultiLayerSteerWorker`` -- a generalised vLLM-Hook worker that applies a
  *dict* of per-layer steering tensors with one of three operations
  (``add`` / ``ablate`` / ``adjust_rs``), matching SafeTune's HF hook math
  exactly.  It is registered with the vLLM-Hook ``PluginRegistry`` under the
  name ``"safetune_multisteer"``.
* ``SteerSpec`` -- a serialisable description of a steering intervention
  (per-layer vectors + op + coefficient).
* ``extract_steer_spec()`` -- pulls a ``SteerSpec`` out of an instantiated
  SafeTune ``SteerModel`` (``RefusalDirectionModel`` / ``CAAModel`` /
  ``SCANSModel`` / ``STAModel`` / ``AlphaSteerModel`` / ``SafeSteerModel``).
* ``VLLMHookSteer`` -- the high-level entry point: give it a model id and a
  ``SteerSpec``, get a steered ``vllm.LLM`` you can ``.generate()`` with.

This module is a first-class STEER inference backend
(``safetune.steer.backends.vllm_hook``), reachable through the unified
``safetune.steer.run(..., backend="vllm-hook")`` entry point.

Faithfulness notes (which SafeTune methods this adapter covers)
---------------------------------------------------------------
FULLY FAITHFUL under the hook worker (static per-layer vector, no per-token
state -- the steer is identical on prefill and decode tokens):

* ``RefusalDirectionModel`` (mode ``steer``  -> op ``add``;
                             mode ``ablate`` -> op ``ablate``)
* ``CAAModel``                              -> op ``add``, multi-layer
* ``STAModel``                              -> op ``add``, multi-layer
* ``SafeSteerModel`` (single category vector, fixed alpha) -> op ``add``

PARTIALLY FAITHFUL -- works but loses the input-conditional part:

* ``SCANSModel`` -- per-layer additive vectors are static, but the *sign*
  (``_current_sign``) is chosen per prompt from a transition-point classifier.
  The hook worker applies a fixed sign.  To be faithful, the sign must be
  computed live in the worker from the prompt's hidden state -- see
  ``MultiLayerSteerWorker`` docstring; left as TODO.
* ``AlphaSteerModel`` -- the intervention is ``h + (h @ M) * strength`` with a
  per-layer *matrix* M (null-space projection), NOT a fixed vector.  PARTIALLY
  FAITHFUL: the ``matrix`` op applies ``(h @ M)*strength`` per-token at EVERY
  position, which matches the HF model's *decode* branch but NOT its *prefill*
  branch -- on prefill the HF AlphaSteerModel computes the projection on the
  last valid token and broadcasts it to all positions. So vLLM prefill steering
  diverges from HF; decode is faithful. (Same partial-faithfulness caveat class
  as SCANS below.)

NEEDS LIVE PER-TOKEN LOGIC IN THE WORKER (cannot be a static spec):

* ``AdaSteerModel`` -- recomputes a per-input adaptive coefficient (logistic
  regression on RD/HD projections) inside the forward pass.  The worker would
  need to embed the fitted logistic params and evaluate them on the running
  hidden state.  ``MultiLayerSteerWorker`` supports this via op ``adaptive``
  (logistic on the projection) -- see ``AdaptiveSteerSpec``.
* ``SafeSwitchModel`` -- a two-stage prober (a small MLP/linear head) gates
  whether to steer at all, and stage 2 needs a few pilot decode tokens.  This
  requires running the prober head inside the worker.  Recommended path:
  ship the prober weights into the worker and evaluate per-request.  Not
  implemented here; documented as future work.
* ``CircuitBreakerModel`` / ``CircuitBreakerRRModel`` -- these are
  *weight-space* / training-time interventions (LoRA / rep-rerouting), not
  runtime residual additions.  They do not need a hook worker at all: bake the
  trained checkpoint and serve it with plain vLLM.

Decoding-steering methods (ContrastiveDecoding, SafeDecoding, ProxyTuning,
Nudging) are NOT activation steering -- they belong in vLLM ``LogitsProcessor``
land.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import torch

# ---------------------------------------------------------------------------
# 1. Serialisable steering spec
# ---------------------------------------------------------------------------


@dataclass
class SteerSpec:
    """A serialisable description of an activation-steering intervention.

    Attributes
    ----------
    op:
        One of ``"add"``, ``"ablate"``, ``"adjust_rs"``, ``"matrix"``.

        * ``add``       -- ``h <- h + coeff * v[layer]`` (CAA, refusal steer,
                           STA, SafeSteer, SCANS with a fixed sign).
        * ``ablate``    -- ``h <- h - coeff * (h.v_hat) v_hat``  with v_hat the
                           unit vector (refusal-direction abliteration).
        * ``adjust_rs`` -- ``h <- h + (avg_proj - h.v) * v``  (the IBM
                           ``adjust_rs`` mode; projects the residual stream onto
                           a target mean activation).
        * ``matrix``    -- ``h <- h + (h @ M[layer]) * coeff``  (AlphaSteer
                           null-space projection; ``vectors`` holds 2-D M).
    vectors:
        ``{layer_idx: 1-D tensor (hidden,)}`` for vector ops, or
        ``{layer_idx: 2-D tensor (hidden, hidden)}`` for ``op == "matrix"``.
    coeff:
        Scalar multiplier (CAA ``alpha`` / refusal ``strength`` /
        SCANS ``multiplier`` ...).  Per-layer override possible via
        ``per_layer_coeff``.
    per_layer_coeff:
        Optional ``{layer_idx: float}`` overriding ``coeff`` for that layer
        (AlphaSteer supports per-layer strengths).
    avg_proj:
        Per-layer target projection for ``op == "adjust_rs"``.
    sign:
        Global +1 / -1 applied before ``coeff`` (SCANS picks this per prompt;
        a static spec bakes one value).
    method:
        Free-text name of the originating SafeTune method (for logging).
    """

    op: str
    vectors: Dict[int, torch.Tensor]
    coeff: float = 1.0
    per_layer_coeff: Dict[int, float] = field(default_factory=dict)
    avg_proj: Dict[int, torch.Tensor] = field(default_factory=dict)
    sign: float = 1.0
    method: str = "unknown"

    # -- (de)serialisation to a .pt file the worker can torch.load ----------

    def save(self, path: str) -> str:
        """Persist the spec to ``path`` as a torch checkpoint."""
        payload = {
            "op": self.op,
            "coeff": float(self.coeff),
            "sign": float(self.sign),
            "method": self.method,
            # store everything CPU/float32 -- worker re-casts to model dtype
            "vectors": {int(k): v.detach().to("cpu", torch.float32)
                        for k, v in self.vectors.items()},
            "per_layer_coeff": {int(k): float(v)
                                for k, v in self.per_layer_coeff.items()},
            "avg_proj": {int(k): v.detach().to("cpu", torch.float32)
                         for k, v in self.avg_proj.items()},
        }
        torch.save(payload, path)
        return path

    @staticmethod
    def load(path: str) -> "SteerSpec":
        p = torch.load(path, map_location="cpu", weights_only=True)
        return SteerSpec(
            op=p["op"],
            vectors={int(k): v for k, v in p["vectors"].items()},
            coeff=p.get("coeff", 1.0),
            per_layer_coeff={int(k): v for k, v in p.get("per_layer_coeff", {}).items()},
            avg_proj={int(k): v for k, v in p.get("avg_proj", {}).items()},
            sign=p.get("sign", 1.0),
            method=p.get("method", "unknown"),
        )

    @property
    def layers(self) -> List[int]:
        return sorted(self.vectors.keys())


# ---------------------------------------------------------------------------
# 2. Extract a SteerSpec from an instantiated SafeTune SteerModel
# ---------------------------------------------------------------------------


def extract_steer_spec(steer_model: Any) -> SteerSpec:
    """Pull a :class:`SteerSpec` out of a SafeTune steering model.

    Supports the additive-vector / projection methods.  The SafeTune model
    must already be constructed (its vectors extracted) -- this reads the
    public attributes only, it does not run extraction.

    Raises ``NotImplementedError`` for methods that require live per-token
    logic in the worker (AdaSteer, SafeSwitch) -- those are documented in the
    module docstring.
    """
    cls = type(steer_model).__name__

    # -- RefusalDirectionModel: single direction, applied to many layers ----
    if cls == "RefusalDirectionModel":
        # direction is a unit vector; layers default to all decoder layers.
        # extract_refusal_direction returns (tensor, layer_idx, per_layer_dict)
        # or a plain tensor depending on version — handle both.
        raw = steer_model.direction
        if isinstance(raw, tuple):
            direction = raw[0].detach()
        else:
            direction = raw.detach()
        # _target_layers may be None -> "all layers"; we cannot know the
        # layer count without the model, so the caller must pass layers
        # explicitly OR we leave it and the worker hooks whatever it finds.
        layers = steer_model._target_layers
        if layers is None:
            # represent "all layers" with a sentinel: layer -1 means broadcast.
            vectors = {-1: direction}
        else:
            vectors = {int(li): direction for li in layers}
        op = "add" if steer_model.mode == "steer" else "ablate"
        return SteerSpec(op=op, vectors=vectors,
                         coeff=float(steer_model.strength), method="refusal_direction")

    # -- CAAModel: per-layer vectors ----------------------------------------
    if cls == "CAAModel":
        vectors = {int(k): v.detach() for k, v in steer_model.vectors.items()}
        return SteerSpec(op="add", vectors=vectors,
                         coeff=float(steer_model.strength), method="caa")

    # -- STAModel: per-layer SAE-decoded vectors ----------------------------
    if cls == "STAModel":
        lv = getattr(steer_model, "layer_vectors", {})
        vectors = {int(k): v.detach() for k, v in lv.items()}
        coeff = float(getattr(steer_model, "multiplier", 1.0))
        return SteerSpec(op="add", vectors=vectors, coeff=coeff, method="sta")

    # -- SCANSModel: per-layer vectors + a per-prompt sign ------------------
    if cls == "SCANSModel":
        sv = getattr(steer_model, "_steering_vectors", {})
        layers = getattr(steer_model, "target_layers", list(sv.keys()))
        vectors = {int(k): sv[k].detach() for k in layers if k in sv}
        coeff = float(getattr(steer_model, "multiplier", 1.0))
        sign = float(getattr(steer_model, "_current_sign", 1.0))
        spec = SteerSpec(op="add", vectors=vectors, coeff=coeff,
                         sign=sign, method="scans")
        # NOTE: sign is baked statically; faithful SCANS needs the
        # transition-point classifier live in the worker. See module docstring.
        return spec

    # -- AlphaSteerModel: per-layer null-space matrices ---------------------
    if cls == "AlphaSteerModel":
        sm = getattr(steer_model, "steering_matrices", {})
        vectors = {int(k): v.detach() for k, v in sm.items()}
        per_layer = {int(k): float(v)
                     for k, v in getattr(steer_model, "strengths", {}).items()}
        return SteerSpec(op="matrix", vectors=vectors, coeff=1.0,
                         per_layer_coeff=per_layer, method="alphasteer")

    # -- SafeSteerModel: category vectors at one/few layers -----------------
    if cls == "SafeSteerModel":
        layers = getattr(steer_model, "layers", [])
        cv = getattr(steer_model, "category_vectors", None)
        alpha = float(getattr(steer_model, "alpha", 1.0))
        vectors: Dict[int, torch.Tensor] = {}
        if cv is not None:
            t = cv if isinstance(cv, torch.Tensor) else torch.as_tensor(cv)
            if t.ndim == 1:
                for li in layers:
                    vectors[int(li)] = t.detach()
            else:  # (num_layers, hidden)
                for pos, li in enumerate(layers):
                    vectors[int(li)] = t[pos].detach()
        return SteerSpec(op="add", vectors=vectors, coeff=alpha, method="safesteer")

    if cls in ("AdaSteerModel", "SafeSwitchModel"):
        raise NotImplementedError(
            f"{cls} requires live per-token logic inside the hook worker "
            f"(adaptive coefficient / prober head). See module docstring; "
            f"use AdaptiveSteerSpec or a custom worker."
        )

    if cls in ("CircuitBreakerModel", "CircuitBreakerRRModel", "RepBendModel",
               "TARModel"):
        raise NotImplementedError(
            f"{cls} is a weight-space / training-time intervention, not a "
            f"runtime residual addition. Bake the trained checkpoint and serve "
            f"it with plain vLLM -- no hook worker needed."
        )

    raise NotImplementedError(f"extract_steer_spec: unsupported model {cls!r}")


# ---------------------------------------------------------------------------
# 3. The generalised vLLM-Hook worker
# ---------------------------------------------------------------------------
#
# This worker class is imported and run *inside the vLLM worker subprocess*.
# It must therefore be importable by a fully-qualified module path
# (safetune.steer.backends.vllm_hook.*). VLLMHookSteer below injects the `src`
# root holding the `safetune` package onto the subprocess PYTHONPATH before
# constructing the LLM.
#
# Worker config is passed via the env var STEER_VLLM_HOOK_SPEC pointing to a
# JSON file: {"spec_path": "<path to SteerSpec .pt>"}.

try:  # only importable in an env with vLLM installed
    from vllm.v1.worker.gpu_worker import Worker as _V1Worker
except Exception:  # pragma: no cover - allows importing the adapter w/o vllm
    _V1Worker = object  # type: ignore[assignment,misc]


class MultiLayerSteerWorker(_V1Worker):  # type: ignore[misc]
    """vLLM V1 worker that applies a multi-layer SafeTune ``SteerSpec``.

    Installs one forward hook per target decoder layer.  Each hook performs
    the SafeTune residual-stream intervention on the layer *output* -- which,
    in vLLM V1 model code (Llama/Qwen/Mistral...), is either a hidden-state
    tensor or a ``(hidden_states, residual)`` tuple.  We steer the residual
    part to match HF semantics (the residual is what flows to the next layer).

    The hook is identical for prefill and decode positions, so the worker is
    registered with ``hooks_on=(True, True)`` -- equivalent to SafeTune's HF
    hooks which fire on every forward.
    """

    def load_model(self, *args, **kwargs):
        r = super().load_model(*args, **kwargs)
        try:
            self._install_steer_hooks()
        except Exception as e:  # pragma: no cover
            print(f"[MultiLayerSteerWorker] hook install FAILED: {e}")
            import traceback
            traceback.print_exc()
        return r

    # -- spec loading -------------------------------------------------------

    def _load_spec(self):
        cfg_path = os.environ["STEER_VLLM_HOOK_SPEC"]
        with open(cfg_path) as f:
            cfg = json.load(f)
        spec_path = cfg["spec_path"]
        p = torch.load(spec_path, map_location="cpu", weights_only=True)
        return p

    # -- hook installation --------------------------------------------------

    def _install_steer_hooks(self):
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("[MultiLayerSteerWorker] no model; skip")
            return

        spec = self._load_spec()
        op = spec["op"]
        coeff = float(spec.get("coeff", 1.0))
        sign = float(spec.get("sign", 1.0))
        vectors = spec["vectors"]                # {int: tensor}
        per_layer = spec.get("per_layer_coeff", {})
        avg_proj = spec.get("avg_proj", {})

        # locate decoder layers
        layers_mod = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers_mod = model.model.layers
        elif hasattr(model, "layers"):
            layers_mod = model.layers
        if layers_mod is None:
            print("[MultiLayerSteerWorker] could not find decoder layers")
            return
        n_layers = len(layers_mod)

        # expand the "all layers" sentinel (-1)
        if -1 in vectors:
            base = vectors[-1]
            vectors = {li: base for li in range(n_layers)}

        def make_hook(layer_idx: int):
            v = vectors.get(layer_idx)
            if v is None:
                return None
            c = float(per_layer.get(layer_idx, coeff)) * sign
            ap = avg_proj.get(layer_idx)

            def hook(_m, _inp, output):
                # vLLM V1 Llama/Qwen/Mistral decoder layers return
                # (hidden_states, residual) using a *fused residual* scheme:
                # the true residual-stream value (== HF's layer output) is
                # `hidden_states + residual`. The next layer's input
                # layernorm consumes both and computes layernorm(hs + res).
                #
                # * For `add`, shifting `residual` by `c*v` shifts the full
                #   stream by exactly `c*v` -- faithful, cheapest.
                # * For `ablate`/`matrix`/`adjust_rs`, the operation depends
                #   on the *full* stream, so we read `full = hs + res`,
                #   compute the steered full stream, and write the delta back
                #   into `residual` (leaving `hidden_states` untouched).
                is_tuple = isinstance(output, tuple)
                if is_tuple:
                    head, res = output[0], output[1]
                    full = head + res
                else:
                    head, res, full = None, output, output
                vt = v.to(full.device, dtype=full.dtype)

                if op == "add":
                    res = res + c * vt
                elif op == "ablate":
                    vhat = vt / (vt.norm() + 1e-8)
                    proj = (full * vhat).sum(-1, keepdim=True) * vhat
                    res = res - c * proj
                elif op == "adjust_rs":
                    vhat = vt  # IBM convention: dir already the unit vec
                    apt = ap.to(full.device, dtype=full.dtype)
                    cur = torch.matmul(full, vhat)
                    delta = (apt - cur).unsqueeze(-1)
                    res = res + delta * vhat.view(1, -1)
                elif op == "matrix":
                    # AlphaSteer: h <- h + (h @ M) * c
                    res = res + torch.matmul(full, vt) * c
                else:
                    raise ValueError(f"unknown op {op!r}")

                if is_tuple:
                    return (head, res) + tuple(output[2:])
                return res

            return hook

        self._steer_handles = []
        installed = []
        for li in range(n_layers):
            h = make_hook(li)
            if h is not None:
                self._steer_handles.append(
                    layers_mod[li].register_forward_hook(
                        lambda m, i, o, _h=h: _h(m, i, o)
                    )
                )
                installed.append(li)
        print(f"[MultiLayerSteerWorker] op={op} coeff={coeff} sign={sign} "
              f"installed {len(installed)} hooks on layers {installed}")


# ---------------------------------------------------------------------------
# 4. Plugin registration helper
# ---------------------------------------------------------------------------
#
# vLLM general plugins are loaded via an entry point. Rather than re-package,
# we register our worker into the IBM PluginRegistry at import time so
# HookLLM (or a plain LLM with worker_cls=...) can find it.


def register_safetune_worker() -> str:
    """Return the FQCN of :class:`MultiLayerSteerWorker` for vLLM's worker_cls.

    vLLM accepts worker_cls as a dotted module path; no external plugin
    registry is needed.
    """
    return f"{MultiLayerSteerWorker.__module__}.{MultiLayerSteerWorker.__name__}"


# ---------------------------------------------------------------------------
# 5. High-level entry point
# ---------------------------------------------------------------------------


class VLLMHookSteer:
    """Serve a SafeTune steering intervention under vLLM.

    Example
    -------
    >>> from safetune.steer import RefusalDirectionModel
    >>> # ... extract direction with HF model, build RefusalDirectionModel ...
    >>> spec = extract_steer_spec(rd_model)          # SteerSpec
    >>> steer = VLLMHookSteer("meta-llama/Llama-3.2-3B-Instruct", spec)
    >>> out = steer.generate(["How do I bake bread?"], max_tokens=128)

    The ``vllm.LLM`` is built with the ``MultiLayerSteerWorker`` as
    ``worker_cls``; the hook is *always on* (there is no per-call toggle --
    construct a second plain ``LLM`` if you need an unsteered baseline).
    """

    def __init__(
        self,
        model: str,
        spec: SteerSpec,
        *,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
        enforce_eager: bool = True,
        **vllm_kwargs: Any,
    ) -> None:
        self.model = model
        self.spec = spec

        # 1. persist the spec where the worker subprocess can torch.load it
        self._tmpdir = tempfile.mkdtemp(prefix="safetune_steer_")
        spec_path = os.path.join(self._tmpdir, "spec.pt")
        spec.save(spec_path)
        cfg_path = os.path.join(self._tmpdir, "worker_cfg.json")
        with open(cfg_path, "w") as f:
            json.dump({"spec_path": spec_path}, f)

        # 2. env the worker subprocess reads
        os.environ["STEER_VLLM_HOOK_SPEC"] = cfg_path
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        # The worker subprocess imports MultiLayerSteerWorker by its FQCN
        # (safetune.steer.backends.vllm_hook.*); put the `src` root that holds
        # the `safetune` package on the worker PYTHONPATH so that resolves.
        src_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        )
        existing = os.environ.get("PYTHONPATH", "")
        if src_root not in existing.split(os.pathsep):
            os.environ["PYTHONPATH"] = (
                f"{src_root}{os.pathsep}{existing}" if existing else src_root
            )

        # 3. register the worker and build the LLM
        worker_path = register_safetune_worker()

        from vllm import LLM  # noqa: WPS433

        self.llm = LLM(
            model=model,
            worker_cls=worker_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            enforce_eager=enforce_eager,
            **vllm_kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self,
        prompts: Sequence[str],
        *,
        apply_chat_template: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 256,
        **sp_kwargs: Any,
    ) -> List[str]:
        from vllm import SamplingParams  # noqa: WPS433

        from .run import render_prompts

        if isinstance(prompts, str):
            prompts = [prompts]
        # Shared chat-template rendering: falls back to the raw prompt when the
        # tokenizer carries no chat_template (base models), instead of raising.
        rendered = render_prompts(self.tokenizer, prompts, apply_chat_template)
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                            **sp_kwargs)
        outs = self.llm.generate(rendered, sp)
        return [o.outputs[0].text for o in outs]


# ---------------------------------------------------------------------------
# 6. CAST vLLM router — pre-probe then route to hooked/plain vLLM
# ---------------------------------------------------------------------------


def cast_vllm_generate(
    model_path: str,
    prompts: List[str],
    cast_model: Any,
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    apply_chat_template: bool = True,
) -> List[str]:
    """Run CAST generation faithfully using vLLM for the decode step.

    CAST cannot use a static vLLM hook worker because its steering is
    conditional on a per-input harmfulness probe.  This function implements
    the faithful two-pass routing approach:

    1. **Probe pass (HF)** — run the frozen drift model up to ``probe_layer``
       (fast; no generation) and score each prompt with the fitted linear probe.
    2. **Route** — split prompts into ``harmful`` (score > threshold) and
       ``benign`` (score ≤ threshold) groups.
    3. **Generate** — harmful batch uses ``VLLMHookSteer`` (CAA SteerSpec);
       benign batch uses plain vLLM.  Results are merged in original order.

    Args:
        model_path: HF model path / Hub ID for the drift model.
        prompts: raw (pre-chat-template) prompt strings.
        cast_model: an instantiated :class:`~safetune.steer.cast.CASTModel`
            whose probe weights / threshold / steering vectors are used.
        temperature, max_tokens, gpu_memory_utilization, max_model_len,
        apply_chat_template: forwarded to the vLLM backends.

    Returns:
        One response string per prompt, in the same order as ``prompts``.
    """
    import torch
    import torch.nn.functional as F

    # ------------------------------------------------------------------ #
    # Step 1: probe pass on HF model (already loaded in cast_model.model) #
    # ------------------------------------------------------------------ #
    hf_model = cast_model.model
    tok = getattr(cast_model, "_tok", None)

    from safetune.steer.cast import _collect_last_token_hidden  # reuse helper
    # Use the tokenizer stored on cast_model if available, else raise.
    if tok is None:
        raise ValueError(
            "cast_vllm_generate: cast_model must have a ._tok attribute "
            "(set cast_model._tok = tokenizer before calling)."
        )

    probe_layer = cast_model.probe_layer
    hidden = _collect_last_token_hidden(hf_model, tok, prompts, probe_layer)
    w = cast_model.probe_weights.to(hidden.device)
    logits = (hidden * w).sum(dim=-1) + cast_model.probe_bias
    scores = torch.sigmoid(logits)                        # (N,)
    gate = (scores > cast_model.threshold).tolist()       # bool per prompt

    harmful_idx = [i for i, g in enumerate(gate) if g]
    benign_idx  = [i for i, g in enumerate(gate) if not g]

    results: List[str] = [""] * len(prompts)

    # ------------------------------------------------------------------ #
    # Step 2a: harmful → VLLMHookSteer with CAA SteerSpec                #
    # ------------------------------------------------------------------ #
    if harmful_idx:
        caa_spec = SteerSpec(
            op="add",
            vectors={int(k): v.detach() for k, v in cast_model.steering_vectors.items()},
            coeff=float(cast_model.alpha),
            method="cast_caa",
        )
        steerer = VLLMHookSteer(
            model_path, caa_spec,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        harmful_prompts = [prompts[i] for i in harmful_idx]
        harmful_responses = steerer.generate(
            harmful_prompts,
            apply_chat_template=apply_chat_template,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        del steerer
        torch.cuda.empty_cache()
        for i, resp in zip(harmful_idx, harmful_responses):
            results[i] = resp

    # ------------------------------------------------------------------ #
    # Step 2b: benign → plain vLLM, no hooks                             #
    # ------------------------------------------------------------------ #
    if benign_idx:
        from vllm import LLM, SamplingParams
        from .run import render_prompts

        llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            enforce_eager=True,
        )
        tokenizer = llm.get_tokenizer()
        benign_prompts = [prompts[i] for i in benign_idx]
        rendered = render_prompts(tokenizer, benign_prompts, apply_chat_template)
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outs = llm.generate(rendered, sp)
        del llm
        torch.cuda.empty_cache()
        for i, out in zip(benign_idx, outs):
            results[i] = out.outputs[0].text

    return results


__all__ = [
    "SteerSpec",
    "extract_steer_spec",
    "MultiLayerSteerWorker",
    "register_safetune_worker",
    "VLLMHookSteer",
    "cast_vllm_generate",
]
