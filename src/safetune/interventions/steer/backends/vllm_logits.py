"""
Adapter: run SafeTune *decoding*-steering methods inside vLLM via the V1
batch-level ``LogitsProcessor`` API.

Background
----------
SafeTune's STEER pillar (eval-design Part 3) has two sub-families:

* **Activation steering** -- residual-stream hooks. Covered by the sibling
  adapter ``steer_vllm_hook.py`` (IBM/vLLM-Hook worker).
* **Decoding steering** -- logit-level interventions at each decode step.
  ``safetune.steer.decoding`` ships four of these as
  ``transformers.LogitsProcessor`` subclasses:

  - ``ContrastiveDecodingProcessor`` -- ``(1+a)*target - a*amateur``
  - ``ProxyTuningProcessor``         -- ``target + scale*(expert - antiexpert)``
  - ``SafeDecodingProcessor``        -- constant-alpha expert blend, first m steps
  - ``NudgingProcessor``             -- defer to a guide when target top-1 prob low

This module wraps those four as vLLM-runnable logits processors.

The hard part: a second model's logits
--------------------------------------
Contrastive / Proxy / Nudging each need a *second* model's next-token logits
(amateur / expert / antiexpert / guide). vLLM runs exactly one model per
engine, so the auxiliary forward cannot come from the same engine.

vLLM 0.11.0 (V1) gives two extension points for logit shaping:

1. ``SamplingParams.logits_processors`` (per-request callables) -- **rejected**
   by V1: ``processor.py:_validate_supported_sampling_params`` raises
   ``"vLLM V1 does not support per request user provided logits processors"``.
   So the HF-style ``(input_ids, scores) -> scores`` path is unavailable.

2. ``LLM(logits_processors=[<class or FQCN>])`` -- the V1 **batch-level**
   ``LogitsProcessor`` (``vllm.v1.sample.logits_processor.interface``). The
   class is instantiated *once inside the worker subprocess* with
   ``(vllm_config, device, is_pin_memory)``; ``update_state(batch_update)``
   feeds it the prompt + running output token ids per request, and
   ``apply(logits)`` reshapes the ``(num_reqs, vocab)`` logit batch right
   before sampling.

Chosen approach for the second model
------------------------------------
Run the auxiliary as a **lightweight HF ``AutoModelForCausalLM`` loaded inside
the same worker subprocess** as the vLLM engine, NOT as a second vLLM engine.

Rationale:

* The batch-level ``LogitsProcessor`` already executes inside the worker
  process and has the worker's CUDA device -- an HF model loaded there shares
  the GPU directly, no IPC, no cross-process logprob plumbing.
* A second ``vllm.LLM`` cannot be constructed inside a worker subprocess
  (nested engine), and constructing it in the *driver* process and querying it
  per step would mean a full driver<->worker round-trip per decode token --
  far slower than a local HF forward.
* The auxiliary models in all four methods are *small* (amateur / expert /
  proxy / guide). A 1-3B HF model alongside a 3B vLLM engine fits A100 80GB
  comfortably.
* It is exactly faithful: the SafeTune processors themselves call the
  auxiliary as a plain HF ``nn.Module`` (see
  ``safetune.steer.decoding._base.GuidedLogitsProcessor._guide_logits``). We
  reuse SafeTune's own ``combine()`` math verbatim, only swapping the *source*
  of the target/aux logits.

Per-step auxiliary forward is ``use_cache=False`` (a full re-forward of the
sequence each step) -- this is precisely what SafeTune's HF processors do
(``_base.GuidedLogitsProcessor._guide_logits`` passes ``use_cache=False``), so
the numerics match. For ContrastiveDecoding / SafeDecoding / Nudging the
*target* logits also come straight from vLLM, so only the aux model is
re-forwarded; vLLM's own KV cache handles the target side at full speed.

Faithfulness summary
--------------------
* ``ContrastiveDecodingProcessor`` -- FULLY FAITHFUL. Per-step, stateless in
  the target; ``combine`` reused verbatim.
* ``ProxyTuningProcessor``         -- FULLY FAITHFUL. Needs TWO aux models
  (expert + antiexpert); both loaded in-process.
* ``SafeDecodingProcessor``        -- FULLY FAITHFUL. The ``first_m`` window is
  derived from ``len(output_tok_ids)`` per request (vLLM gives the running
  output list by reference).
* ``NudgingProcessor``             -- FAITHFUL to the per-step decision rule
  (defer to guide when target top-1 prob < threshold). The paper's phrase-level
  hand-off grouping is, as the SafeTune docstring already notes, not
  expressible in a per-step logit hook -- same constraint as the HF backend,
  not a vLLM-specific loss.

This module is a first-class STEER inference backend
(``safetune.steer.backends.vllm_logits``), reachable through the unified
``safetune.steer.run(..., backend="vllm-logits")`` entry point.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

import torch

# --------------------------------------------------------------------------
# Make SafeTune importable both here (driver) and in the worker subprocess.
# --------------------------------------------------------------------------
# `src` root that holds the `safetune` package (this file is
# src/safetune/steer/backends/vllm_logits.py -> four levels up).
_SAFETUNE_SRC = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..")
)
if _SAFETUNE_SRC not in sys.path:
    sys.path.insert(0, _SAFETUNE_SRC)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ===========================================================================
# 1. Serialisable spec describing one decoding-steering intervention
# ===========================================================================


@dataclass
class DecodeSteerSpec:
    """Serialisable description of a decoding-steering intervention.

    Round-trips through a JSON file the worker subprocess reads (the worker
    cannot receive Python objects directly).

    Attributes
    ----------
    method:
        One of ``"contrastive"``, ``"proxy_tuning"``, ``"safedecoding"``,
        ``"nudging"``.
    aux_model:
        HF id / path of the primary auxiliary model:

        * contrastive  -> the *amateur* (weak) model
        * proxy_tuning -> the *expert* (proxy-tuned small) model
        * safedecoding -> the *expert* (safety-tuned) model
        * nudging      -> the *guide* (small aligned) model
    aux_model_2:
        Second auxiliary (``proxy_tuning`` only -- the *antiexpert* / proxy-base
        small model). ``None`` for the other three.
    aux_dtype:
        Torch dtype for the auxiliary HF model(s).
    params:
        Method-specific hyper-parameters (mirror the SafeTune ``*Config``
        dataclasses): e.g. ``{"alpha": 0.5, "adaptive_eps": 0.1}``.
    """

    method: str
    aux_model: str
    aux_model_2: Optional[str] = None
    aux_dtype: str = "bfloat16"
    params: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "DecodeSteerSpec":
        d = json.loads(s)
        return DecodeSteerSpec(**d)

    def save(self, path: str) -> str:
        with open(path, "w") as f:
            f.write(self.to_json())
        return path

    @staticmethod
    def load(path: str) -> "DecodeSteerSpec":
        with open(path) as f:
            return DecodeSteerSpec.from_json(f.read())


# Env var the worker subprocess reads to find its DecodeSteerSpec JSON file.
_SPEC_ENV = "SAFETUNE_DECODE_STEER_SPEC"


# ===========================================================================
# 2. The vLLM V1 batch-level LogitsProcessor
# ===========================================================================
#
# vLLM imports this class by FQCN inside the worker subprocess and instantiates
# it once with (vllm_config, device, is_pin_memory). It must therefore be
# importable by module path -- VLLMDecodeSteer below injects this dir onto the
# worker PYTHONPATH (mirrors steer_vllm_hook.VLLMHookSteer).

try:  # only importable in an env with vLLM installed
    from vllm.v1.sample.logits_processor.interface import (
        BatchUpdate,
        LogitsProcessor as _V1LogitsProcessor,
        MoveDirectionality,
    )
except Exception:  # pragma: no cover - allows importing the adapter w/o vllm
    _V1LogitsProcessor = object  # type: ignore[assignment,misc]
    BatchUpdate = object  # type: ignore[assignment,misc]
    MoveDirectionality = None  # type: ignore[assignment]


class _ReqState:
    """Per-batch-row tracking: prompt ids + a live ref to output token ids."""

    __slots__ = ("prompt_tok_ids", "output_tok_ids")

    def __init__(self, prompt_tok_ids: Optional[List[int]],
                 output_tok_ids: List[int]):
        # vLLM gives prompt_tok_ids possibly None for some flows; coerce.
        self.prompt_tok_ids: List[int] = list(prompt_tok_ids or [])
        # output_tok_ids is a *reference* to the engine's running list -- do
        # NOT copy; it grows in place as tokens are generated.
        self.output_tok_ids: List[int] = output_tok_ids

    def full_ids(self) -> List[int]:
        return self.prompt_tok_ids + list(self.output_tok_ids)

    @property
    def n_generated(self) -> int:
        return len(self.output_tok_ids)


class SafeTuneDecodeLogitsProcessor(_V1LogitsProcessor):  # type: ignore[misc]
    """vLLM V1 batch-level logits processor wrapping a SafeTune decoding method.

    One instance lives for the engine's lifetime inside the worker subprocess.
    It loads the auxiliary HF model(s) once, tracks each batch row's token
    sequence via :meth:`update_state`, and in :meth:`apply` re-forwards the
    auxiliary model(s) to obtain the second-model logits, then delegates the
    blend to the SafeTune processor's own ``combine()`` math.
    """

    # ----- vLLM-mandated constructor ---------------------------------------
    def __init__(self, vllm_config, device: torch.device,
                 is_pin_memory: bool) -> None:
        self.device = torch.device(device)
        self._rows: Dict[int, _ReqState] = {}

        spec_path = os.environ.get(_SPEC_ENV)
        if not spec_path:
            raise RuntimeError(
                f"{_SPEC_ENV} not set -- SafeTuneDecodeLogitsProcessor needs a "
                f"DecodeSteerSpec JSON path in the environment.")
        self.spec = DecodeSteerSpec.load(spec_path)
        self.method = self.spec.method

        # vLLM model id (target). Used only to fetch the tokenizer for the
        # SafeTune processors' vocab-translation table.
        self._target_model_id = vllm_config.model_config.model

        self._dtype = getattr(torch, self.spec.aux_dtype)

        # Lazy: built on first apply() (after CUDA is fully initialised in the
        # worker). Loading in __init__ races vLLM's own model load on memory.
        self._built = False
        self._aux = None          # primary aux HF model
        self._aux2 = None         # secondary aux HF model (proxy antiexpert)
        self._st_proc = None      # the SafeTune *Processor instance
        self._tok = None

        # diagnostics
        self._apply_calls = 0
        self._aux_forward_s = 0.0

        print(f"[SafeTuneDecodeLogitsProcessor] method={self.method} "
              f"aux={self.spec.aux_model} aux2={self.spec.aux_model_2} "
              f"device={self.device}")

    # ----- one-time heavy build -------------------------------------------
    def _build(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Tokenizer of the *target* (vLLM) model -- the SafeTune processors
        # build their vocab-translation table from target vs aux tokenizers.
        self._tok = AutoTokenizer.from_pretrained(self._target_model_id)
        aux_tok = AutoTokenizer.from_pretrained(self.spec.aux_model)

        self._aux = AutoModelForCausalLM.from_pretrained(
            self.spec.aux_model, dtype=self._dtype,
        ).to(self.device).eval()

        if self.spec.aux_model_2:
            self._aux2 = AutoModelForCausalLM.from_pretrained(
                self.spec.aux_model_2, dtype=self._dtype,
            ).to(self.device).eval()

        # Instantiate the genuine SafeTune processor -- we reuse its combine()
        # / _common_token_mask() / config math verbatim. We never call its
        # __call__ (that would trigger its own guide forward); we drive
        # combine() directly with logits we supply.
        p = self.spec.params
        if self.method == "contrastive":
            from safetune.steer.decoding import (
                ContrastiveDecodingConfig, ContrastiveDecodingProcessor)
            cfg = ContrastiveDecodingConfig(
                alpha=float(p.get("alpha", 0.5)),
                adaptive_eps=float(p.get("adaptive_eps", 0.1)))
            self._st_proc = ContrastiveDecodingProcessor(
                guide=self._aux, tokenizer_target=self._tok,
                tokenizer_guide=aux_tok, config=cfg)

        elif self.method == "proxy_tuning":
            from safetune.steer.decoding import (
                ProxyTuningConfig, ProxyTuningProcessor)
            cfg = ProxyTuningConfig(
                scale=float(p.get("scale", 1.0)),
                clamp_delta=p.get("clamp_delta"))
            self._st_proc = ProxyTuningProcessor(
                proxy_tuned=self._aux, proxy_base=self._aux2,
                tokenizer_target=self._tok, tokenizer_proxy=aux_tok,
                config=cfg)

        elif self.method == "safedecoding":
            from safetune.steer.decoding import (
                SafeDecodingConfig, SafeDecodingProcessor)
            cfg = SafeDecodingConfig(
                alpha=float(p.get("alpha", 1.0)),
                first_m=int(p.get("first_m", 5)),
                num_common_tokens=int(p.get("num_common_tokens", 3)),
                top_k=int(p.get("top_k", 50)),
                clip_negative_inf=bool(p.get("clip_negative_inf", True)))
            # prompt_length is set per-row dynamically in apply(); pass 0 here.
            self._st_proc = SafeDecodingProcessor(
                guide=self._aux, tokenizer_target=self._tok, prompt_length=0,
                tokenizer_guide=aux_tok, config=cfg)

        elif self.method == "nudging":
            from safetune.steer.decoding import (
                NudgingConfig, NudgingProcessor)
            cfg = NudgingConfig(
                top_prob_thres=float(p.get("top_prob_thres", 0.3)),
                soft_blend=bool(p.get("soft_blend", False)),
                soft_blend_temp=float(p.get("soft_blend_temp", 0.1)))
            self._st_proc = NudgingProcessor(
                guide=self._aux, tokenizer_target=self._tok,
                tokenizer_guide=aux_tok, config=cfg)
        else:
            raise ValueError(f"unknown decoding-steer method {self.method!r}")

        self._built = True
        print(f"[SafeTuneDecodeLogitsProcessor] built {self.method}; "
              f"aux on {self.device}")

    # ----- vLLM hooks ------------------------------------------------------
    def is_argmax_invariant(self) -> bool:
        # All four methods reshape the distribution in a way that can move the
        # argmax (that is the whole point) -- never argmax-invariant.
        return False

    def update_state(self, batch_update: Optional["BatchUpdate"]) -> None:
        """Track prompt + running output token ids per batch row.

        Order mandated by vLLM: removed, added, moved.
        """
        if batch_update is None:
            return

        for idx in batch_update.removed:
            self._rows.pop(idx, None)

        for idx, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            # output_tok_ids is a live reference -- store it as-is.
            self._rows[idx] = _ReqState(prompt_tok_ids, output_tok_ids)

        for adx, bdx, direct in batch_update.moved:
            a = self._rows.get(adx)
            b = self._rows.get(bdx)
            if direct == MoveDirectionality.SWAP:
                if a is not None:
                    self._rows[bdx] = a
                else:
                    self._rows.pop(bdx, None)
                if b is not None:
                    self._rows[adx] = b
                else:
                    self._rows.pop(adx, None)
            else:  # UNIDIRECTIONAL a -> b
                if a is not None:
                    self._rows[bdx] = a
                else:
                    self._rows.pop(bdx, None)
                self._rows.pop(adx, None)

    @torch.no_grad()
    def _aux_last_logits(self, model, ids: List[int]) -> torch.Tensor:
        """Last-step logits of an aux model for a token sequence.

        Mirrors ``GuidedLogitsProcessor._guide_logits``: a full
        ``use_cache=False`` forward, last position only -> ``(1, vocab)``.
        """
        t = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = model(input_ids=t, use_cache=False)
        return out.logits[:, -1, :].float()

    @torch.no_grad()
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Reshape the ``(num_reqs, vocab)`` target-logit batch in place."""
        if not self._built:
            self._build()
        self._apply_calls += 1

        n = logits.shape[0]
        st = self._st_proc
        method = self.method

        for row in range(n):
            state = self._rows.get(row)
            if state is None:
                # No tracking info -- leave the row untouched.
                continue
            ids = state.full_ids()
            if len(ids) == 0:
                continue

            target_row = logits[row:row + 1, :].float()  # (1, vocab)

            t0 = time.perf_counter()
            # --- aux forward(s) -------------------------------------------
            aux_native = self._aux_last_logits(self._aux, ids)  # (1, V_aux)
            self._aux_forward_s += time.perf_counter() - t0

            # Translate aux logits into target-vocab space using the SafeTune
            # processor's own translation table (identity when vocabs match).
            translation = st._target_to_guide
            if translation.numel() == 0:
                guide_scores = aux_native
            else:
                translation = translation.to(aux_native.device)
                mask = translation >= 0
                gathered = aux_native[:, translation.clamp(min=0)]
                neg_inf = torch.full_like(gathered, float("-inf"))
                guide_scores = torch.where(mask, gathered, neg_inf)

            # --- delegate the blend to the SafeTune combine() -------------
            if method == "contrastive" or method == "nudging":
                # combine(scores, guide_scores, input_ids)
                fake_input_ids = torch.tensor([ids], device=self.device)
                new_row = st.combine(target_row, guide_scores, fake_input_ids)

            elif method == "safedecoding":
                # SafeDecoding's combine uses step = input_ids.shape[1] -
                # prompt_length. We set prompt_length so step == n_generated.
                st.prompt_length = len(ids) - state.n_generated
                fake_input_ids = torch.tensor([ids], device=self.device)
                new_row = st.combine(target_row, guide_scores, fake_input_ids)

            elif method == "proxy_tuning":
                # ProxyTuning needs a SECOND aux forward (antiexpert/base).
                # The SafeTune ProxyTuningProcessor.combine() internally calls
                # self._proxy_base_logits(input_ids) on self.proxy_base. That
                # forwards proxy_base on raw input_ids -- which is exactly the
                # aux token sequence here. So we can call combine() directly.
                fake_input_ids = torch.tensor([ids], device=self.device)
                new_row = st.combine(target_row, guide_scores, fake_input_ids)
            else:  # pragma: no cover
                raise ValueError(method)

            logits[row, :] = new_row.to(logits.dtype).squeeze(0)

        return logits

    # ----- diagnostics -----------------------------------------------------
    def stats(self) -> Dict[str, float]:
        return {
            "apply_calls": float(self._apply_calls),
            "aux_forward_s": self._aux_forward_s,
        }


# ===========================================================================
# 3. High-level driver-side entry point
# ===========================================================================


class VLLMDecodeSteer:
    """Serve a SafeTune decoding-steering method under vLLM.

    Example
    -------
    >>> steer = VLLMDecodeSteer(
    ...     target_model="meta-llama/Llama-3.2-3B-Instruct",
    ...     spec=DecodeSteerSpec(
    ...         method="contrastive",
    ...         aux_model="meta-llama/Llama-3.2-1B-Instruct",
    ...         params={"alpha": 0.5, "adaptive_eps": 0.1}))
    >>> out = steer.generate(["How do I bake bread?"], max_tokens=128)

    The ``vllm.LLM`` is built with ``SafeTuneDecodeLogitsProcessor`` registered
    via ``logits_processors=[...]``; the processor is always on (build a plain
    ``LLM`` for an unsteered baseline).
    """

    _FQCN = f"{__name__}:SafeTuneDecodeLogitsProcessor"

    def __init__(
        self,
        target_model: str,
        spec: DecodeSteerSpec,
        *,
        gpu_memory_utilization: float = 0.55,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
        enforce_eager: bool = True,
        **vllm_kwargs: Any,
    ) -> None:
        self.target_model = target_model
        self.spec = spec

        # 1. persist the spec where the worker subprocess can read it
        self._tmpdir = tempfile.mkdtemp(prefix="safetune_decode_steer_")
        spec_path = os.path.join(self._tmpdir, "spec.json")
        spec.save(spec_path)

        # 2. env the worker subprocess reads
        os.environ[_SPEC_ENV] = spec_path
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        # make THIS module + SafeTune importable in the worker subprocess
        for d in (_THIS_DIR, _SAFETUNE_SRC):
            existing = os.environ.get("PYTHONPATH", "")
            if d not in existing.split(os.pathsep):
                os.environ["PYTHONPATH"] = (
                    f"{d}{os.pathsep}{existing}" if existing else d)

        # 3. build the LLM with the custom batch-level logits processor.
        #    Note: gpu_memory_utilization is left low-ish by default so the
        #    auxiliary HF model has room on the same GPU.
        from vllm import LLM  # noqa: WPS433

        # vLLM accepts a logitsproc *class* directly; passing the class avoids
        # an FQCN-resolution round-trip, but the class still has to be
        # importable in the worker (handled by PYTHONPATH above). We pass the
        # FQCN string for robustness across spawn.
        self.llm = LLM(
            model=target_model,
            logits_processors=[self._FQCN],
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


__all__ = [
    "DecodeSteerSpec",
    "SafeTuneDecodeLogitsProcessor",
    "VLLMDecodeSteer",
]
