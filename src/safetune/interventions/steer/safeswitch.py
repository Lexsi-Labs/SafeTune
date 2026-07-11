"""SafeSwitch inference-time prober + refusal-head model wrapper.

Faithful re-implementation of SafeSwitch (Han et al., Findings of EMNLP 2025,
arXiv:2502.01042, https://github.com/Hanpx20/SafeSwitch).

The authors' design has two trained components:

1. A **two-stage prober**.

   * Stage 1 -- *instruction-safety prober* -- reads the last-token hidden
     state of a deep transformer layer in the *prefill* phase and estimates
     ``p_instr``, the probability the prompt carries a harmful instruction.
   * Stage 2 -- *compliance prober* -- after decoding a small number of
     "pilot" tokens (the paper finds 3 near-optimal), reads the last-token
     hidden state of the same layer and estimates ``p_compliance``, the
     probability the model is about to comply with an unsafe request.

   The two stages are combined multiplicatively
   (paper Eq. 2): ``p_unsafe = p_instr * p_compliance``.  Each prober is a
   small MLP (paper / repo ``LinearProber``: ``Linear -> ReLU -> Linear ->
   Softmax``, intermediate dim 64, output dim 2, < 1M params).

2. A **refusal head** -- a fine-tuned copy of the LM head ``T_R`` (a full
   ``|V| x d_model`` weight matrix, ~6% of model params).  At inference, when
   ``p_unsafe`` exceeds the threshold the base LM head is *replaced* by the
   refusal head for the whole generation (paper Eq. 3):

       P(y|x) = softmax( T_R H_L )  if p_unsafe(x) > tau   (head REPLACED)
       P(y|x) = softmax( T   H_L )  otherwise

   The repo stores and substitutes the *whole* trained head weight, so this
   wrapper accepts the refusal head as a complete ``lm_head`` weight tensor
   and swaps it in/out around generation.

This module keeps the public ``SafeSwitchModel`` signature backward
compatible: every new capability is exposed through optional keyword
arguments with defaults that fall back to the previous (logit-bias) behaviour
when no trained refusal head is supplied.
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    from safetune.core.runtime.inference.safeswitch import (
        SafeSwitchConfig as _CoreSafeSwitchConfig,
        SafeSwitchRunner,
        SafetyProber,
    )
    _IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    _CoreSafeSwitchConfig = None  # type: ignore[assignment]
    SafeSwitchRunner = None  # type: ignore[assignment]
    SafetyProber = None  # type: ignore[assignment]
    _IMPORT_ERROR = _e


class MLPProber:
    """Two-layer MLP probe over a single layer's last-token hidden state.

    Mirrors the authors' ``LinearProber`` (SafeSwitch repo ``src/utils.py``):
    ``Linear(d_model, hidden) -> ReLU -> Linear(hidden, 2) -> Softmax``.
    With ``intermediate_dim`` defaulting to 64, as in the paper's Appendix A.

    The prober reads the *last input token* of layer ``layer_idx`` -- the
    paper's last-token pooling ``H_L`` -- not a mean over the sequence.
    """

    def __init__(
        self,
        hidden_size: int,
        layer_idx: int = -1,
        intermediate_dim: int = 64,
    ) -> None:
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.intermediate_dim = intermediate_dim
        self._net = None  # populated by build()/train()

    def build(self) -> None:
        """Instantiate the underlying ``torch`` MLP (lazy; CPU-friendly)."""
        import torch.nn as nn

        if self._net is not None:
            return
        self._net = nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_dim),
            nn.ReLU(),
            nn.Linear(self.intermediate_dim, 2),
        )

    def _last_token_feature(self, hidden_states: Any, attention_mask: Any = None):
        """Pool layer ``layer_idx`` at the last (non-pad) token position."""

        hs = hidden_states[self.layer_idx]  # (batch, seq, hidden)
        if attention_mask is not None:
            # last non-padding index per row
            lengths = attention_mask.long().sum(dim=1) - 1  # (batch,)
            idx = lengths.clamp(min=0).view(-1, 1, 1).expand(-1, 1, hs.size(-1))
            pooled = hs.gather(1, idx).squeeze(1)
        else:
            pooled = hs[:, -1, :]
        return pooled.float()

    def predict_unsafe_probability(
        self, hidden_states: Any, attention_mask: Any = None
    ) -> float:
        """Return ``P(unsafe)`` for the current hidden-state batch."""
        import torch

        if self._net is None:
            raise RuntimeError(
                "MLPProber must be built/trained before use. Call build() "
                "or train() / load weights first."
            )
        feats = self._last_token_feature(hidden_states, attention_mask)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(feats)
            prob = torch.softmax(logits, dim=-1)[0, 1]
        return float(prob)

    def load_state_dict(self, state_dict: Any) -> None:
        self.build()
        self._net.load_state_dict(state_dict)


class TwoStageProber:
    """SafeSwitch's two-stage prober (paper Section 3, Eq. 2).

    Combines an instruction-safety prober (stage 1, prefill) and a compliance
    prober (stage 2, after a few pilot tokens) multiplicatively:
    ``p_unsafe = p_instr * p_compliance``.

    If only a stage-1 prober is provided, stage 2 is skipped and
    ``p_unsafe = p_instr`` -- a graceful degradation to the single-stage probe.
    """

    def __init__(
        self,
        instr_prober: Any,
        compliance_prober: Any = None,
    ) -> None:
        self.instr_prober = instr_prober
        self.compliance_prober = compliance_prober

    def predict_unsafe_probability(
        self, hidden_states: Any, attention_mask: Any = None
    ) -> float:
        p_instr = _prober_prob(self.instr_prober, hidden_states, attention_mask)
        if self.compliance_prober is None:
            return p_instr
        p_comp = _prober_prob(
            self.compliance_prober, hidden_states, attention_mask
        )
        return p_instr * p_comp


def _prober_prob(prober: Any, hidden_states: Any, attention_mask: Any) -> float:
    """Call a prober's probability method, tolerant of signature variants."""
    try:
        return float(
            prober.predict_unsafe_probability(hidden_states, attention_mask)
        )
    except TypeError:
        # core SafetyProber takes hidden_states only
        return float(prober.predict_unsafe_probability(hidden_states))


class SafeSwitchModel:
    """Wrapper exposing SafeSwitch via a standard model API.

    Faithful behaviour (paper Eq. 2-3):

    * Prefill -> stage-1 instruction prober -> ``p_instr``.
    * If a compliance prober is given, decode ``pilot_tokens`` pilot tokens
      and run stage-2 -> ``p_compliance``; combine ``p_unsafe = p_instr *
      p_compliance``.
    * If ``p_unsafe > unsafe_threshold`` (paper default 0.5) generate with the
      *refusal head* substituted for the base LM head; otherwise generate
      normally.

    Backward compatibility: the original signature is preserved.  When no
    ``compliance_prober`` and no ``refusal_head`` are supplied, the wrapper
    falls back to the legacy single-stage probe with the fixed
    ``refusal_logit_bonus`` logit bias on ``refusal_token_ids`` via the core
    :class:`SafeSwitchRunner`.
    """

    def __init__(
        self,
        model: Any,
        prober: Any = None,
        probe_layer: int = -1,
        hidden_size: int = 4096,
        unsafe_threshold: float = 0.5,
        refusal_token_ids: Optional[List[int]] = None,
        refusal_logit_bonus: float = 10.0,
        compliance_prober: Any = None,
        refusal_head: Any = None,
        pilot_tokens: int = 3,
        prober_intermediate_dim: int = 64,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.runtime.inference.safeswitch is unavailable"
            ) from _IMPORT_ERROR
        self.model = model
        self.probe_layer = probe_layer
        self.hidden_size = hidden_size
        self.unsafe_threshold = unsafe_threshold
        self.pilot_tokens = max(0, int(pilot_tokens))

        # --- stage-1 instruction prober -------------------------------------
        # Default to the faithful two-layer MLP prober; callers may still pass
        # any object exposing predict_unsafe_probability (incl. the core
        # sklearn-based SafetyProber) for backward compatibility.
        if prober is None:
            prober = MLPProber(
                hidden_size=hidden_size,
                layer_idx=probe_layer,
                intermediate_dim=prober_intermediate_dim,
            )
        self.instr_prober = prober
        self.compliance_prober = compliance_prober
        self._two_stage = TwoStageProber(prober, compliance_prober)

        # --- refusal head ----------------------------------------------------
        # SafeSwitch's refusal head is a fine-tuned full LM-head weight matrix
        # that *replaces* the base head when the prober fires (paper Eq. 3).
        self.refusal_head = refusal_head
        self._base_head_weight = None  # snapshot for restore

        # --- legacy fallback (logit-bias) -----------------------------------
        # Kept so callers that supplied neither a trained refusal head nor a
        # compliance prober still get the previous behaviour.
        config = _CoreSafeSwitchConfig(
            probe_layer=probe_layer,
            hidden_size=hidden_size,
            unsafe_threshold=unsafe_threshold,
            refusal_token_ids=list(refusal_token_ids or []),
            refusal_logit_bonus=refusal_logit_bonus,
        )
        self._config = config
        # The core runner is only used for the legacy path; build it with a
        # prober it can actually consume.
        _core_prober = prober
        if not hasattr(prober, "predict_unsafe_probability"):
            _core_prober = SafetyProber(
                hidden_size=hidden_size, layer_idx=probe_layer
            )
        self._impl = SafeSwitchRunner(
            model=model, prober=_core_prober, config=config
        )

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _faithful_path_available(self) -> bool:
        """True when we can run the full paper algorithm (refusal-head swap)."""
        return self.refusal_head is not None

    def _lm_head(self) -> Any:
        head = getattr(self.model, "lm_head", None)
        if head is None:
            raise AttributeError(
                "SafeSwitch refusal-head substitution requires the base model "
                "to expose a `lm_head` module."
            )
        return head

    def _prefill_hidden_states(self, input_ids: Any, attention_mask: Any = None):
        import torch

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        return out.hidden_states

    def _pilot_hidden_states(self, input_ids: Any, attention_mask: Any = None):
        """Decode `pilot_tokens` tokens, return hidden states at that point.

        Implements the paper's stage-2 "pilot" decoding: a small number of
        tokens are generated first, then the compliance prober reads the
        last-token hidden state of the extended sequence.
        """
        import torch

        if self.pilot_tokens <= 0:
            return self._prefill_hidden_states(input_ids, attention_mask)
        with torch.no_grad():
            pilot = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.pilot_tokens,
                do_sample=False,
            )
            pilot_mask = None
            if attention_mask is not None:
                extra = pilot.shape[1] - input_ids.shape[1]
                pad = torch.ones(
                    (attention_mask.shape[0], extra),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                pilot_mask = torch.cat([attention_mask, pad], dim=1)
            out = self.model(
                input_ids=pilot,
                attention_mask=pilot_mask,
                output_hidden_states=True,
            )
        return out.hidden_states, pilot_mask

    def _compute_p_unsafe(self, input_ids: Any, attention_mask: Any = None) -> float:
        """Two-stage unsafe probability (paper Eq. 2)."""

        # stage 1 -- instruction prober on prefill
        try:
            hs_prefill = self._prefill_hidden_states(input_ids, attention_mask)
            p_instr = _prober_prob(self.instr_prober, hs_prefill, attention_mask)
        except Exception:
            return 0.0

        if self.compliance_prober is None:
            return p_instr

        # stage 2 -- compliance prober after pilot decoding
        try:
            res = self._pilot_hidden_states(input_ids, attention_mask)
            if isinstance(res, tuple):
                hs_pilot, pilot_mask = res
            else:
                hs_pilot, pilot_mask = res, None
            p_comp = _prober_prob(self.compliance_prober, hs_pilot, pilot_mask)
        except Exception:
            # if stage 2 fails, fall back to stage-1 estimate only
            return p_instr
        return p_instr * p_comp

    def _generate_with_refusal_head(self, input_ids: Any, **kwargs: Any) -> Any:
        """Swap in the refusal head, generate, restore the base head.

        Faithful to the repo: the trained refusal head is a full LM-head
        weight tensor substituted for `model.lm_head.weight`.
        """
        import torch

        head = self._lm_head()
        self._base_head_weight = head.weight.data
        try:
            new_weight = self.refusal_head
            if hasattr(new_weight, "weight"):  # an nn.Linear-like head
                new_weight = new_weight.weight
            new_weight = new_weight.to(
                dtype=head.weight.dtype, device=head.weight.device
            )
            head.weight = torch.nn.Parameter(new_weight, requires_grad=False)
            return self.model.generate(input_ids=input_ids, **kwargs)
        finally:
            # always restore the base head, even on error
            head.weight = torch.nn.Parameter(
                self._base_head_weight, requires_grad=False
            )
            self._base_head_weight = None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        input_ids = None
        if args:
            input_ids = args[0]
        elif "input_ids" in kwargs:
            input_ids = kwargs.pop("input_ids")

        if input_ids is None:
            # nothing to probe -- pass straight through
            return self.model.generate(**kwargs)

        # Legacy path: no trained refusal head -> defer to the core runner,
        # which applies the fixed logit-bias fallback.
        if not self._faithful_path_available():
            return self._impl.generate(input_ids, **kwargs)

        attention_mask = kwargs.get("attention_mask")
        p_unsafe = self._compute_p_unsafe(input_ids, attention_mask)

        if p_unsafe > self.unsafe_threshold:
            return self._generate_with_refusal_head(input_ids, **kwargs)
        return self.model.generate(input_ids=input_ids, **kwargs)

    def predict_unsafe_probability(
        self, input_ids: Any, attention_mask: Any = None
    ) -> float:
        """Expose the two-stage ``p_unsafe`` score for a given prompt."""
        return self._compute_p_unsafe(input_ids, attention_mask)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        return getattr(self.__dict__["model"], name)

    def remove_hooks(self) -> None:
        # SafeSwitch installs no forward hooks; restore the head if a swap was
        # left dangling by an interrupted generation.
        if self._base_head_weight is not None:
            try:
                import torch

                head = self._lm_head()
                head.weight = torch.nn.Parameter(
                    self._base_head_weight, requires_grad=False
                )
            except Exception:  # pragma: no cover
                pass
            self._base_head_weight = None
        return None

    def __enter__(self) -> "SafeSwitchModel":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "SafeSwitchModel":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(path)
        return cls(model, **kwargs)
