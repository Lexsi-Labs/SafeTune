"""SafeSteer inference-time steering model wrapper.

Faithful re-implementation of SafeSteer — "SafeSteer: Interpretable Safety
Steering with Refusal-Evasion in LLMs" (Ghosh et al., arXiv:2506.04250) and its
precursor "Towards Inference-time Category-wise Safety Steering for Large
Language Models" (arXiv:2410.01174).

SafeSteer is a *training-free*, gradient-free, unsupervised activation-steering
method for safe decoding. Its two defining mechanisms — both **missing** from
the old thin re-export of ``runtime/inference/safesteer.py`` — are restored
here:

1. **Median-norm pruning of activation differences.** The steering vector for a
   harm category ``c`` is *not* a plain diff-of-means. The paper computes
   per-example activation differences between unsafe and safe data, takes the
   L2 norm of each difference, computes the **median** of those norms, and
   keeps only the differences whose norm exceeds the median (the top ~50% — the
   most informative signals). The category vector ``omega^(c)`` is the mean of
   the *retained* differences. This discards low-magnitude differences that
   conflate harm features with generic content features.

   Paper eq. (category-wise steering vector):
       omega^(c) = mean_{j in D_safe^(c)} act(x_j^safe)
                 - mean_{j in D_unsafe^(c)} act(x_j^unsafe)
   refined by median-norm pruning over the paired differences.

2. **Multi-layer steering.** SafeSteer intervenes on a *set* of intermediate
   layers (the paper tests ``{14, 16, 20, 25, 31}`` for a 32-layer Llama-2-7B),
   not a single layer. Steering is additive on the self-attention sub-block
   output of each intervention layer:
       theta_l^attn  ->  theta_l^attn + m * omega_l^(c)
   with multiplier ``m`` ~ 0.5-1.0.

The old wrapper did a plain ``mean(safe) - mean(unsafe)`` at a single fixed
``layer_id=20`` with no pruning. Both gaps are closed here without depending on
``runtime/inference/safesteer.py`` (left untouched).

Public-signature compatibility: ``SafeSteerModel.__init__`` keeps its original
parameters in order (``model, category_vectors, classifier, layer_id, alpha,
default_category``). New mechanisms are exposed as **optional keyword args with
defaults** — ``layers``, ``prune``, ``prune_quantile``, ``per_layer_vectors`` —
so ``from safetune.steer import SafeSteerModel`` and existing call sites are
unaffected.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

try:  # torch is the only hard dependency for the steering math
    import torch

    _TORCH_ERROR: Optional[Exception] = None
except Exception as _te:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_ERROR = _te


# --------------------------------------------------------------------------
# Steering-vector extraction with median-norm pruning (paper §3, eq. omega^(c))
# --------------------------------------------------------------------------
def prune_by_median_norm(
    differences: "torch.Tensor",
    quantile: float = 0.5,
) -> "torch.Tensor":
    """Median-norm pruning of per-example activation differences.

    ``differences`` is a ``(N, hidden)`` tensor of per-example
    ``act(safe) - act(unsafe)`` vectors. Computes the L2 norm of each row, the
    ``quantile``-th quantile (default the **median**, 0.5) of those norms, and
    returns only the rows whose norm **exceeds** the threshold — i.e. the most
    informative top-(1-quantile) fraction. Faithful to the paper's
    "keep only those differences whose norms exceed the median (top 50%)".

    If pruning would discard everything (e.g. all norms equal), the unpruned
    tensor is returned so the vector is never empty.
    """
    if differences.ndim != 2:
        differences = differences.reshape(differences.shape[0], -1)
    norms = differences.norm(dim=1)
    if norms.numel() == 0:
        return differences
    thresh = torch.quantile(norms.float(), quantile)
    keep = norms > thresh
    if not bool(keep.any()):
        # all norms <= threshold (degenerate / ties) — keep everything
        return differences
    return differences[keep]


def compute_category_vectors(
    safe_acts: Dict[str, "torch.Tensor"],
    unsafe_acts: Dict[str, "torch.Tensor"],
    prune: bool = True,
    prune_quantile: float = 0.5,
) -> Dict[str, "torch.Tensor"]:
    """Compute a SafeSteer steering vector per harm category.

    ``safe_acts[c]`` / ``unsafe_acts[c]`` are ``(N, hidden)`` activation
    matrices (one row per example, already token-averaged) for category ``c``.

    With ``prune=True`` (the paper's default) the steering vector is the mean
    of the **median-norm-pruned** paired differences. To form pairs the two
    matrices are truncated to a common length ``min(N_safe, N_unsafe)``; if the
    counts differ wildly the diff-of-means form is used as a fallback so no
    signal is silently dropped.

    With ``prune=False`` this reduces to the plain diff-of-means
    ``mean(safe) - mean(unsafe)`` (the old behavior), kept for callers that
    pre-pruned their data.
    """
    vectors: Dict[str, "torch.Tensor"] = {}
    for cat in safe_acts.keys():
        if cat not in unsafe_acts:
            continue
        s = safe_acts[cat]
        u = unsafe_acts[cat]
        if s.ndim == 1:
            s = s[None, :]
        if u.ndim == 1:
            u = u[None, :]
        if not prune:
            vectors[cat] = s.mean(dim=0) - u.mean(dim=0)
            continue
        n = min(s.shape[0], u.shape[0])
        if n >= 2:
            diffs = s[:n] - u[:n]
            pruned = prune_by_median_norm(diffs, quantile=prune_quantile)
            vectors[cat] = pruned.mean(dim=0)
        else:
            # not enough paired examples to estimate a median — diff-of-means
            vectors[cat] = s.mean(dim=0) - u.mean(dim=0)
    return vectors


class SafeSteerModel:
    """SafeSteer: category-routed, multi-layer, training-free activation steering.

    Parameters
    ----------
    model:
        A causal-LM (HF ``transformers``-style) whose decoder layers expose a
        ``model.model.layers`` or ``model.transformer.h`` list.
    category_vectors:
        ``{category -> steering tensor}``. A tensor may be:
          * a single ``(hidden,)`` vector — applied (as-is) to every layer in
            ``layers``; or
          * a ``(num_layers, hidden)`` matrix — row ``i`` applied to the i-th
            layer in ``layers`` (per-layer vectors, the paper's preferred form
            since ``omega`` is computed and applied at the *same* layer).
        Pass ``per_layer_vectors=True`` to force per-layer interpretation.
    classifier:
        Optional ``prompt_text -> category`` router. ``None`` always routes to
        ``default_category``.
    layer_id:
        Kept for backward compatibility. When ``layers`` is not given, the
        intervention layer set defaults to ``[layer_id]`` (old single-layer
        behavior).
    alpha:
        Steering multiplier ``m`` (paper uses ~0.5-1.0).
    default_category:
        Category used when no classifier is set or routing fails.
    layers:
        **Optional.** Explicit set of intervention layer indices. The paper
        steers a set of intermediate layers (e.g. ``{14, 16, 20, 25, 31}`` for
        a 32-layer model). When omitted, falls back to ``[layer_id]``.
    prune / prune_quantile:
        **Optional.** Carried for callers using the bundled
        ``compute_category_vectors`` classmethod (median-norm pruning); they do
        not affect a model constructed with pre-computed ``category_vectors``.
    per_layer_vectors:
        **Optional.** Force per-layer interpretation of 2-D category tensors.
    """

    # expose the pruning-aware extractor as a classmethod (paper-faithful)
    compute_category_vectors = staticmethod(compute_category_vectors)
    prune_by_median_norm = staticmethod(prune_by_median_norm)

    def __init__(
        self,
        model: Any,
        category_vectors: Dict[str, Any],
        classifier: Optional[Callable[[str], str]] = None,
        layer_id: int = 20,
        alpha: float = 1.0,
        default_category: str = "default",
        *,
        layers: Optional[Sequence[int]] = None,
        prune: bool = True,
        prune_quantile: float = 0.5,
        per_layer_vectors: bool = False,
    ) -> None:
        if _TORCH_ERROR is not None:  # pragma: no cover
            raise ImportError("SafeSteerModel requires torch") from _TORCH_ERROR

        self.model = model
        self.category_vectors: Dict[str, Any] = dict(category_vectors)
        self.classifier = classifier
        self.layer_id = int(layer_id)
        self.alpha = float(alpha)
        self.default_category = default_category
        self.prune = bool(prune)
        self.prune_quantile = float(prune_quantile)
        self.per_layer_vectors = bool(per_layer_vectors)

        # Multi-layer intervention set (paper §3 — a set of intermediate
        # layers). Backward-compatible default: the single legacy layer.
        if layers is None:
            self.layers: List[int] = [self.layer_id]
        else:
            self.layers = [int(x) for x in layers]
        if not self.layers:
            raise ValueError("SafeSteerModel: 'layers' must be non-empty")

        self._hooks: List[Any] = []
        self._current_category: Optional[str] = None

        self.register_hooks()

    # ------------------------------------------------------------------
    # Layer access
    # ------------------------------------------------------------------
    def _decoder_layers(self) -> Any:
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        if hasattr(m, "layers"):
            return m.layers
        raise AttributeError("SafeSteerModel: could not locate transformer layers")

    # ------------------------------------------------------------------
    # Steering-vector selection: per-layer (paper) or shared
    # ------------------------------------------------------------------
    def _vector_for(self, category: str, layer_pos: int) -> Optional["torch.Tensor"]:
        """Return the steering vector for ``category`` at the ``layer_pos``-th
        intervention layer, or ``None`` if unavailable.

        A ``(num_layers, hidden)`` category tensor is treated as per-layer
        (the paper computes/applies ``omega_l`` at the *same* layer ``l``); a
        ``(hidden,)`` tensor is shared across all intervention layers.
        """
        if category not in self.category_vectors:
            return None
        v = self.category_vectors[category]
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v)
        if v.ndim == 2:
            # per-layer matrix: row i -> i-th intervention layer
            if v.shape[0] == len(self.layers) or self.per_layer_vectors:
                idx = min(layer_pos, v.shape[0] - 1)
                return v[idx]
            # 2-D but not aligned to layer set — collapse to its mean vector
            return v.mean(dim=0)
        return v

    # ------------------------------------------------------------------
    # Hooks — additive steering on each intervention layer
    # ------------------------------------------------------------------
    def _make_hook(self, layer_pos: int) -> Callable:
        def hook(module: Any, inputs: Any, output: Any) -> Any:
            cat = self._current_category or self.default_category
            v = self._vector_for(cat, layer_pos)
            if v is None:
                return output
            if isinstance(output, tuple):
                hidden = output[0]
                add = (self.alpha * v).to(hidden.device).to(hidden.dtype)
                hidden = hidden + add.reshape(*([1] * (hidden.ndim - 1)), -1)
                return (hidden,) + tuple(output[1:])
            hidden = output
            add = (self.alpha * v).to(hidden.device).to(hidden.dtype)
            return hidden + add.reshape(*([1] * (hidden.ndim - 1)), -1)

        return hook

    def register_hooks(self) -> None:
        self.remove_hooks()
        decoder = self._decoder_layers()
        n = len(decoder)
        for pos, layer_idx in enumerate(self.layers):
            if not (0 <= layer_idx < n):
                raise IndexError(
                    f"SafeSteerModel: layer index {layer_idx} out of range "
                    f"[0, {n})"
                )
            h = decoder[layer_idx].register_forward_hook(self._make_hook(pos))
            self._hooks.append(h)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Category routing
    # ------------------------------------------------------------------
    def set_current_prompt(self, prompt_text: str) -> None:
        if self.classifier is None:
            self._current_category = self.default_category
            return
        try:
            self._current_category = self.classifier(prompt_text)
        except Exception:
            self._current_category = self.default_category

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        input_ids: Any = None,
        prompt_text: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        if prompt_text is not None:
            self.set_current_prompt(prompt_text)
        if input_ids is not None:
            return self.model.generate(input_ids, **kwargs)
        return self.model.generate(**kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        return getattr(self.__dict__["model"], name)

    def __enter__(self) -> "SafeSteerModel":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "SafeSteerModel":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(path)
        return cls(model, **kwargs)
