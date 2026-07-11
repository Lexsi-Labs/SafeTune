"""SCANS inference-time activation steering model wrapper.

Faithful re-implementation of:

    "SCANS: Mitigating the Exaggerated Safety for LLMs via Safety-Conscious
    Activation Steering", Cao, Yang, Zhao. AAAI 2025. arXiv:2408.11491.
    Original repo: https://github.com/zouyingcao/SCANS

This module is implemented *locally* (it is no longer a thin re-export of
``safetune.core.runtime.inference.scans``) so that the two defining mechanisms of
SCANS are restored faithfully:

1. **Vocabulary-projection layer selection** (paper §3.2). The refusal
   steering vector of every layer is projected into vocabulary space through
   the LM head; layers whose top-projected tokens carry refusal concepts are
   the "safety-critical" layers used for steering. Following the authors, the
   layer index is divided into three uniform segments (former / middle /
   latter) and the *middle* segment is anchored — the authors note "our
   steering performance is insensitive to the choice of specific layers ...
   provided they are within the middle layers".

2. **Transition-based adaptive sign** (paper §3.3, repo ``SCANS_llama.py``).
   For each prompt ``q`` the hidden-state transition
   ``aₜˡ(q) = aₚˡ(q+r_pos) − aₑˡ(q+r_pos)`` is compared (cosine similarity,
   averaged over a layer set ``L``) against a reference harm direction
   ``d_harmˡ`` extracted from the harmful anchor set. The per-prompt sign is

       σ(q) = +1  if  s_q ≥ T   (harmful  -> steer *toward* refusal)
       σ(q) = -1  if  s_q <  T   (benign   -> steer *away* from refusal)

   and the residual stream is modified by ``ãˡ = aˡ + σ(q)·α·vᵣˡ`` (Eq. 6).
   This is the whole point of SCANS: a *fixed* sign would raise refusal on
   benign prompts, the exact over-refusal SCANS exists to remove.

Public-signature compatibility: ``SCANSModel.__init__`` keeps the original
five parameters (``model``, ``steering_vectors``, ``target_layers``,
``multiplier``, ``anchor_size``); the new mechanisms are wired through
*optional* keyword arguments with defaults so callers in ``steer/__init__``
are unaffected.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from tqdm.auto import tqdm

try:  # pragma: no cover - torch is a hard runtime dep, optional at import
    import torch
    import torch.nn.functional as F
    _TORCH_ERR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_ERR = _e


# ---------------------------------------------------------------------------
# Model-structure helpers
# ---------------------------------------------------------------------------

def _get_decoder_layers(model: Any) -> List[Any]:
    """Return the list of transformer decoder blocks for common architectures."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
        return list(inner.language_model.layers)
    if inner is not None and hasattr(inner, "layers"):
        return list(inner.layers)
    tr = getattr(model, "transformer", None)
    if tr is not None and hasattr(tr, "h"):
        return list(tr.h)
    if hasattr(model, "layers"):
        return list(model.layers)
    return []


def _num_layers(model: Any) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "num_hidden_layers", None):
        return int(cfg.num_hidden_layers)
    return len(_get_decoder_layers(model))


def _get_lm_head(model: Any) -> Optional[Any]:
    """Return the unembedding (LM head) module/weight, if discoverable."""
    for attr in ("lm_head", "output", "embed_out"):
        head = getattr(model, attr, None)
        if head is not None:
            return head
    return None


def _get_final_norm(model: Any) -> Optional[Any]:
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "norm"):
        return inner.norm
    tr = getattr(model, "transformer", None)
    if tr is not None and hasattr(tr, "ln_f"):
        return tr.ln_f
    return None


# ---------------------------------------------------------------------------
# Refusal token vocabulary (used to score vocabulary projections, paper §3.2 /
# Table 1: middle-layer projections promote tokens such as "cannot", "reject",
# "impossible").
# ---------------------------------------------------------------------------

_REFUSAL_TOKEN_HINTS: Sequence[str] = (
    "cannot", "reject", "rejected", "impossible", "refuse", "refused",
    "sorry", "apolog", "unable", "illegal", "harmful", "unsafe",
    "inappropriate", "decline", "won", "against",
)


class SCANSModel:
    """Wrapper that applies SCANS safety-conscious activation steering.

    SCANS performs *adaptive-sign* refusal steering at the model's
    safety-critical (middle) layers. Steering vectors and the reference harm
    direction can either be precomputed and passed in, or computed on the fly
    from anchor prompts via :meth:`fit`.
    """

    # Positive-response continuation appended to a query when extracting the
    # hidden-state transition (paper §3.3 / repo: ``q + ' Sure'``).
    _POSITIVE_RESPONSE = "Sure"

    def __init__(
        self,
        model: Any,
        steering_vectors: Optional[Dict[int, Any]] = None,
        target_layers: Optional[List[int]] = None,
        multiplier: float = 3.5,
        anchor_size: int = 64,
        *,
        reference_harm: Optional[Dict[int, Any]] = None,
        classification_layers: Optional[List[int]] = None,
        threshold: float = 0.75,
        normalize_vectors: bool = True,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Construct a SCANS steering wrapper.

        Args:
            model: a causal-LM (``transformers``-style) module.
            steering_vectors: optional precomputed ``{layer: vᵣˡ}`` refusal
                steering vectors (Eq. 1). If omitted, call :meth:`fit`.
            target_layers: optional explicit safety-critical layer indices.
                When ``None`` they are derived by vocabulary projection in
                :meth:`fit`, or default to the middle third of the model.
            multiplier: steering strength ``α`` (Eq. 6). Paper default 3.5.
            anchor_size: size of the anchor set used by :meth:`fit`.
            reference_harm: optional precomputed ``{layer: d_harmˡ}`` reference
                transition directions (Eq. 3) used for adaptive-sign
                classification.
            classification_layers: layer set ``L`` over which the transition
                cosine similarity is averaged (Eq. 4). Defaults to the middle
                and latter layers.
            threshold: classification threshold ``T`` (Eq. 5). Paper uses
                0.65-0.75 depending on dataset; 0.75 is the default.
            normalize_vectors: unit-normalize each steering vector (matches the
                authors' ``steering_v /= torch.norm(steering_v)``).
            tokenizer: tokenizer for :meth:`fit` / :meth:`predict_safety`.
                Optional if vectors are supplied precomputed.
        """
        if _TORCH_ERR is not None:  # pragma: no cover
            raise ImportError("SCANSModel requires PyTorch") from _TORCH_ERR

        self.model = model
        self.tokenizer = tokenizer
        self.multiplier = float(multiplier)
        self.anchor_size = int(anchor_size)
        self.threshold = float(threshold)
        self.normalize_vectors = bool(normalize_vectors)

        n_layers = _num_layers(model)
        self._n_layers = n_layers

        # Safety-critical layers: explicit > default middle third.
        if target_layers is not None:
            self.target_layers: List[int] = list(target_layers)
        else:
            self.target_layers = self._default_middle_segment(n_layers)

        # Classification layer set L (middle + latter layers, paper §3.3).
        if classification_layers is not None:
            self.classification_layers: List[int] = list(classification_layers)
        else:
            self.classification_layers = list(
                range(n_layers // 3, n_layers)
            )

        self._steering_vectors: Dict[int, Any] = {}
        if steering_vectors is not None:
            self.set_precomputed_vectors(steering_vectors)

        self._reference_harm: Dict[int, Any] = dict(reference_harm or {})

        # Per-prompt steering sign σ(q); +1 -> toward refusal, -1 -> away.
        self._current_sign: float = 1.0
        self._hooks: List[Any] = []

        # If vectors were supplied up front, arm the hooks immediately so the
        # wrapper behaves like the previous version (which registered hooks in
        # __init__ when given precomputed vectors).
        if self._steering_vectors:
            self.register_hooks()

    # ------------------------------------------------------------------
    # Layer-selection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_middle_segment(n_layers: int) -> List[int]:
        """Middle third of layer indices (paper's three-part division)."""
        if n_layers <= 0:
            return list(range(10, 20))
        lo = n_layers // 3
        hi = (2 * n_layers) // 3
        if hi <= lo:  # tiny models -> at least one layer
            hi = min(n_layers, lo + 1)
        return list(range(lo, hi))

    def select_safety_critical_layers(
        self,
        steering_vectors: Optional[Dict[int, Any]] = None,
    ) -> List[int]:
        """Anchor safety-critical layers via vocabulary projection (paper §3.2).

        Each layer's refusal steering vector ``vᵣˡ`` is projected into
        vocabulary space through the (final-norm + LM-head) unembedding. A
        layer is "safety-critical" when its top-projected tokens contain
        refusal concepts (``cannot``, ``reject``, ``impossible``, ...). The
        authors divide layers into former / middle / latter thirds and pick the
        segment with the strongest refusal projection — empirically always the
        middle third.

        Falls back to the middle-third heuristic when the LM head cannot be
        located or the tokenizer is unavailable.
        """
        vecs = steering_vectors if steering_vectors is not None else self._steering_vectors
        n = self._n_layers
        former = list(range(0, n // 3))
        middle = list(range(n // 3, (2 * n) // 3))
        latter = list(range((2 * n) // 3, n))

        head = _get_lm_head(self.model)
        norm = _get_final_norm(self.model)
        if not vecs or head is None or self.tokenizer is None:
            return middle or self._default_middle_segment(n)

        # Build the unembedding weight matrix W_U: [vocab, hidden].
        w_u = getattr(head, "weight", None)
        if w_u is None and torch is not None and isinstance(head, torch.Tensor):
            w_u = head
        if w_u is None:
            return middle or self._default_middle_segment(n)

        # Precompute refusal-token vocabulary ids once.
        refusal_ids = self._refusal_token_ids()
        if not refusal_ids:
            return middle or self._default_middle_segment(n)

        def segment_refusal_score(layer_ids: List[int]) -> float:
            """Mean fraction of top-k projected tokens that are refusal tokens."""
            scores: List[float] = []
            with torch.no_grad():
                for li in layer_ids:
                    if li not in vecs:
                        continue
                    v = vecs[li].float()
                    v = v.to(w_u.device)
                    if norm is not None:
                        try:
                            v = norm(v.to(next(p.dtype for p in norm.parameters())))
                            v = v.float()
                        except Exception:
                            pass  # un-normalizable param-free norm -> skip
                    logits = torch.matmul(v, w_u.float().t())  # [vocab]
                    topk = torch.topk(logits, k=min(20, logits.numel())).indices
                    topk_set = set(int(t) for t in topk.tolist())
                    hit = len(topk_set & refusal_ids) / max(1, len(topk_set))
                    scores.append(hit)
            return sum(scores) / len(scores) if scores else 0.0

        seg_scores = {
            "former": (former, segment_refusal_score(former)),
            "middle": (middle, segment_refusal_score(middle)),
            "latter": (latter, segment_refusal_score(latter)),
        }
        best = max(seg_scores.values(), key=lambda kv: kv[1])
        chosen = best[0]
        return chosen or middle or self._default_middle_segment(n)

    def _refusal_token_ids(self) -> set:
        ids: set = set()
        tok = self.tokenizer
        if tok is None:
            return ids
        for hint in _REFUSAL_TOKEN_HINTS:
            for variant in (hint, " " + hint, hint.capitalize(), " " + hint.capitalize()):
                try:
                    enc = tok.encode(variant, add_special_tokens=False)
                except Exception:
                    continue
                if len(enc) == 1:
                    ids.add(int(enc[0]))
        return ids

    # ------------------------------------------------------------------
    # Vector extraction (paper Eqs. 1-3)
    # ------------------------------------------------------------------

    def _hidden_states(self, text: str) -> Any:
        """Run a forward pass and return stacked per-layer hidden states.

        Returns a tensor of shape ``[n_layers+1, seq, hidden]`` (index 0 is the
        embedding output, index ``l+1`` is decoder layer ``l``'s output).
        """
        tok = self.tokenizer
        if tok is None:
            raise ValueError("a tokenizer is required to extract activations")
        enc = tok(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        kwargs: Dict[str, Any] = {"output_hidden_states": True, "return_dict": True}
        if attn is not None:
            kwargs["attention_mask"] = attn.to(device)
        with torch.no_grad():
            out = self.model(input_ids, **kwargs)
        # [n_layers+1, batch, seq, hidden] -> drop batch dim 0.
        hs = torch.stack(out.hidden_states)[:, 0, :, :].detach().float().cpu()
        return hs

    def compute_vectors(
        self,
        harmful_prompts: Sequence[str],
        harmless_prompts: Sequence[str],
    ) -> Dict[int, Any]:
        """Extract per-layer refusal steering vectors ``vᵣˡ`` (Eq. 1).

        ``vᵣˡ = mean_{q⁻} aˡ(q⁻) − mean_{q⁺} aˡ(q⁺)`` at the last token
        position — the direction *from answering toward refusal*.
        """
        harm_hs = [
            self._hidden_states(t)
            for t in tqdm(harmful_prompts, desc="SCANS [harmful]", unit="prompt", leave=False)
        ]
        safe_hs = [
            self._hidden_states(t)
            for t in tqdm(harmless_prompts, desc="SCANS [harmless]", unit="prompt", leave=False)
        ]
        vectors: Dict[int, Any] = {}
        for layer in range(self._n_layers):
            li = layer + 1  # hidden_states index (0 = embeddings)
            neg = torch.stack([h[li, -1] for h in harm_hs])
            pos = torch.stack([h[li, -1] for h in safe_hs])
            v = neg.mean(dim=0) - pos.mean(dim=0)
            if self.normalize_vectors:
                nrm = torch.norm(v)
                if nrm > 0:
                    v = v / nrm
            vectors[layer] = v
        return vectors

    def _transition_vectors(self, text_with_pos: str) -> Any:
        """Per-layer hidden-state transition ``aₜˡ = aₚˡ − aₑˡ`` (Eq. 2).

        ``aₑˡ`` is the final-token activation of ``q + r_pos``; ``aₚˡ`` is the
        activation at the last token of the query part ``q`` (i.e. one position
        before ``r_pos``). The text passed in already has ``r_pos`` appended.
        """
        hs = self._hidden_states(text_with_pos)  # [n_layers+1, seq, hidden]
        # r_pos ("Sure") is appended as the final token(s); aₚ is the position
        # immediately before it, aₑ is the final position.
        trans: Dict[int, Any] = {}
        for layer in range(self._n_layers):
            li = layer + 1
            a_e = hs[li, -1]
            a_p = hs[li, -2] if hs.shape[1] >= 2 else hs[li, -1]
            trans[layer] = a_p - a_e
        return trans

    def compute_reference_harm(self, harmful_prompts: Sequence[str]) -> Dict[int, Any]:
        """Reference harm direction ``d_harmˡ`` (Eq. 3).

        Average of the hidden-state transition over the harmful anchor set.
        """
        per_prompt: List[Dict[int, Any]] = []
        for q in harmful_prompts:
            text = f"{q} {self._POSITIVE_RESPONSE}"
            per_prompt.append(self._transition_vectors(text))
        ref: Dict[int, Any] = {}
        for layer in range(self._n_layers):
            stacked = torch.stack([p[layer] for p in per_prompt])
            ref[layer] = stacked.mean(dim=0)
        return ref

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        harmful_prompts: Sequence[str],
        harmless_prompts: Sequence[str],
    ) -> "SCANSModel":
        """Run the full SCANS preparation stage on anchor data.

        Computes (1) refusal steering vectors, (2) safety-critical layers via
        vocabulary projection, (3) the reference harm direction for adaptive
        sign classification. Anchor sets are truncated to ``anchor_size``.
        """
        harmful = list(harmful_prompts)[: self.anchor_size]
        harmless = list(harmless_prompts)[: self.anchor_size]

        self._steering_vectors = self.compute_vectors(harmful, harmless)
        self.target_layers = self.select_safety_critical_layers(self._steering_vectors)
        self._reference_harm = self.compute_reference_harm(harmful)
        self.register_hooks()
        return self

    def set_precomputed_vectors(self, vectors: Dict[int, Any]) -> None:
        """Load precomputed refusal steering vectors ``{layer: vᵣˡ}``."""
        self._steering_vectors = {}
        for layer, v in vectors.items():
            v = v.float() if hasattr(v, "float") else v
            if self.normalize_vectors and hasattr(v, "norm"):
                nrm = torch.norm(v)
                if nrm > 0:
                    v = v / nrm
            self._steering_vectors[int(layer)] = v

    def set_reference_harm(self, reference: Dict[int, Any]) -> None:
        """Load precomputed reference harm directions ``{layer: d_harmˡ}``."""
        self._reference_harm = {int(k): v for k, v in reference.items()}

    # ------------------------------------------------------------------
    # Adaptive-sign classification (paper §3.3, Eqs. 4-5)
    # ------------------------------------------------------------------

    def predict_safety(self, prompt: str) -> tuple:
        """Classify a prompt and return ``(similarity, sign)``.

        Computes the transition cosine similarity ``s_q`` averaged over the
        classification layers (Eq. 4) and the steering sign ``σ(q)`` (Eq. 5):
        ``σ = +1`` (steer toward refusal) when ``s_q ≥ T``, else ``σ = -1``
        (steer away from refusal, the over-refusal fix).
        """
        if not self._reference_harm:
            # No reference available -> default to refusal-steering sign.
            return 1.0, 1.0
        text = f"{prompt} {self._POSITIVE_RESPONSE}"
        trans = self._transition_vectors(text)
        cosines: List[float] = []
        for layer in self.classification_layers:
            if layer not in trans or layer not in self._reference_harm:
                continue
            a_t = trans[layer].unsqueeze(0)
            d_harm = self._reference_harm[layer].float().unsqueeze(0)
            cos = torch.nan_to_num(F.cosine_similarity(a_t, d_harm))
            cosines.append(float(cos.item()))
        if not cosines:
            return 1.0, 1.0
        s_q = sum(cosines) / len(cosines)
        sign = 1.0 if s_q >= self.threshold else -1.0
        return s_q, sign

    def set_prompt(self, prompt: str) -> float:
        """Classify ``prompt`` and arm the per-prompt steering sign σ(q).

        Call this *before* :meth:`generate` so the hooks apply the correct,
        input-dependent sign. Returns the chosen sign.
        """
        _, sign = self.predict_safety(prompt)
        self._current_sign = float(sign)
        return self._current_sign

    # ------------------------------------------------------------------
    # Hooks (Eq. 6: ãˡ = aˡ + σ(q)·α·vᵣˡ)
    # ------------------------------------------------------------------

    def _hook_fn(self, layer_idx: int):
        def hook(module: Any, inputs: Any, output: Any) -> Any:
            if layer_idx not in self._steering_vectors:
                return output
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            v = self._steering_vectors[layer_idx].to(hidden.device).to(hidden.dtype)
            steered = hidden + self._current_sign * self.multiplier * v
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered

        return hook

    def register_hooks(self) -> None:
        """Install forward hooks on the anchored safety-critical layers."""
        self.remove_hooks()
        layers = _get_decoder_layers(self.model)
        for idx in self.target_layers:
            if 0 <= idx < len(layers) and idx in self._steering_vectors:
                self._hooks.append(
                    layers[idx].register_forward_hook(self._hook_fn(idx))
                )

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, *args: Any, prompt: Optional[str] = None, **kwargs: Any) -> Any:
        """Generate with SCANS steering.

        If ``prompt`` is supplied, the adaptive sign σ(q) is recomputed for it
        before generation (the SCANS classification step). Otherwise the
        currently armed sign is used.
        """
        if prompt is not None:
            self.set_prompt(prompt)
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        return getattr(self.__dict__["model"], name)

    def __enter__(self) -> "SCANSModel":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "SCANSModel":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(path)
        kwargs.setdefault("tokenizer", AutoTokenizer.from_pretrained(path))
        return cls(model, **kwargs)
