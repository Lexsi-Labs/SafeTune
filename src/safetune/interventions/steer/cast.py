"""
CAST: Conditional Activation Steering (Wu et al., arXiv:2409.05907, ICLR 2025).

"Programming Refusal with Conditional Activation Steering" — IBM, 2024/2025.
Reference implementation: https://github.com/IBM/activation-steering
(``activation_steering/malleable_model.py`` + ``leash_layer.py``).

CAST extends CAA (Panickssery et al., arXiv:2312.06681) with a *conditioning*
mechanism so that the behavior (refusal) steering vector is added **only when a
condition fires**, leaving unrelated inputs untouched. The defining contribution
is the **condition vector + cosine-similarity gate** (NOT a logistic probe):

  1. A **condition vector** is the (normalized) difference-of-means of hidden
     states for *condition-present* (harmful) vs *condition-absent* (benign)
     prompts at a **condition layer** — exactly the same geometry as a CAA
     contrast vector, but used for *detection* rather than *intervention*.

  2. At inference, at the **prefill (prompt) pass**, the pooled hidden state
     ``h`` at the condition layer is compared to the condition vector ``v`` by
     **cosine similarity** of ``h`` with its projection onto ``v``
     (``proj = tanh(P·h)`` where ``P = vᵀv / (v·v)``;
     ``sim = cos(h, proj)``, matching ``leash_layer.compute_similarity``).
     The gate fires when ``sim`` crosses a **threshold** under a learned
     comparator direction (``condition_comparator_threshold_is``).

  3. The (layer, threshold, comparator-direction) triple is chosen by a
     **GRID SEARCH** that maximizes the **condition F1** of firing on the
     target (harmful) category but not others, on the contrast set
     (``malleable_model.find_best_condition_point``).

  4. When the gate fires, the standard **CAA behavior vector** (difference-of-
     means of the behavior contrast) is added to the residual stream at the
     behavior layers; otherwise the forward pass is completely unmodified.

This conditional gating enables *per-category surgical steering* — e.g. a
bioweapons refusal vector fires only when the bioweapons condition triggers,
leaving unrelated queries fully unaffected. The default ``alpha`` here is ~1.0
(the paper applies modest behavior strengths, not the ~20 used by some other
unconditional methods).

Comparator semantics (faithful to the IBM repo, which is deliberately inverted):
  * ``"larger"``  → condition met when ``sim <  threshold``
  * ``"smaller"`` → condition met when ``sim >  threshold``
We expose a clearer alias too: ``fire_when`` ∈ {``"greater"``, ``"less"``}
maps to ``sim > threshold`` / ``sim < threshold`` respectively.

The condition vector + grid-searched threshold are produced by
:func:`fit_cast_condition`.

Reference: B. Wu, S. Iyer, H. Yao, K. Chen, et al., "Programming Refusal with
Conditional Activation Steering." arXiv:2409.05907 (2024). Library:
https://github.com/IBM/activation-steering (Apache-2.0).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from safetune._refusal_helpers import _get_decoder_layers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Condition-vector fitting (faithful CAST)
# ---------------------------------------------------------------------------

@dataclass
class CASTCondition:
    """A fitted CAST condition (cosine-similarity gate).

    Attributes:
        condition_vector: ``(hidden,)`` L2-normalized difference-of-means
            condition direction at ``condition_layer`` (condition-present minus
            condition-absent).
        condition_layer: decoder-block index at which the gate is checked.
        threshold: cosine-similarity threshold chosen by grid search.
        comparator: ``"larger"`` or ``"smaller"`` (IBM-repo semantics). The
            condition fires when:
              ``"larger"``  → ``sim < threshold``
              ``"smaller"`` → ``sim > threshold``
        f1: condition F1 achieved by the chosen point on the fit set.
        pool: ``"mean"`` (default, paper) or ``"last"`` token pooling.
    """

    condition_vector: torch.Tensor
    condition_layer: int
    threshold: float
    comparator: str = "smaller"
    f1: float = 0.0
    pool: str = "mean"


def _cast_similarity(h: torch.Tensor, condition_vector: torch.Tensor) -> float:
    """Cosine-similarity condition score (faithful to ``leash_layer``).

    Builds the rank-1 projector ``P = vᵀv / (v·v)``, projects the pooled hidden
    state ``proj = tanh(P·h)``, and returns ``cos(h, proj)`` — exactly the
    quantity compared to the threshold in IBM/activation-steering.
    """
    h = h.flatten().float()
    v = condition_vector.flatten().float().to(h.device)
    vv = torch.dot(v, v).clamp_min(1e-12)
    # P·h = v * (v·h) / (v·v)  — avoids materializing the (d,d) outer product.
    proj = torch.tanh(v * (torch.dot(v, h) / vv))
    denom = (h.norm() * proj.norm()).clamp_min(1e-12)
    return float(torch.dot(h, proj) / denom)


def _condition_met(sim: float, threshold: float, comparator: str) -> bool:
    """Apply the (inverted) IBM comparator semantics."""
    if comparator == "smaller":
        return sim > threshold
    if comparator == "larger":
        return sim < threshold
    raise ValueError(f"comparator must be 'larger' or 'smaller', got {comparator!r}")


def _pool_hidden(h_seq: torch.Tensor, pool: str) -> torch.Tensor:
    """Pool a ``(seq, hidden)`` prefill activation to ``(hidden,)``."""
    if pool == "mean":
        return h_seq.mean(dim=0)
    if pool == "last":
        return h_seq[-1, :]
    raise ValueError(f"pool must be 'mean' or 'last', got {pool!r}")


def _collect_pooled_hidden(
    model: nn.Module,
    tokenizer: Any,
    prompts: List[str],
    layer_indices: List[int],
    pool: str = "mean",
    batch_size: int = 8,
) -> Dict[int, torch.Tensor]:
    """Collect pooled hidden states at several layers for a list of prompts.

    Returns ``{layer_idx: (N, hidden) float32 CPU tensor}``. Prompts are run one
    at a time (batch_size=1) so per-prompt pooling matches inference (where the
    gate sees a single sequence at prefill).
    """
    layers = _get_decoder_layers(model)
    for li in layer_indices:
        if not (0 <= li < len(layers)):
            raise IndexError(
                f"fit_cast_condition: condition layer {li} out of range "
                f"(model has {len(layers)} decoder layers)."
            )

    captured: Dict[int, torch.Tensor] = {}

    def _mk_hook(li: int):
        def _hook(_m: nn.Module, _i: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            # h: (batch, seq, hidden). batch is 1 here.
            captured[li] = _pool_hidden(h[0].detach().float().cpu(), pool)
        return _hook

    handles = [layers[li].register_forward_hook(_mk_hook(li)) for li in layer_indices]
    out: Dict[int, List[torch.Tensor]] = {li: [] for li in layer_indices}
    try:
        model.eval()
        device = next(model.parameters()).device
        for p in prompts:
            captured.clear()
            enc = tokenizer([p], return_tensors="pt", padding=True, truncation=True)
            if hasattr(enc, "to"):
                enc = enc.to(device)
            else:
                enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in dict(enc).items()}
            with torch.no_grad():
                model(**enc)
            for li in layer_indices:
                if li in captured:
                    out[li].append(captured[li])
    finally:
        for hd in handles:
            hd.remove()

    result: Dict[int, torch.Tensor] = {}
    for li in layer_indices:
        if not out[li]:
            raise RuntimeError(f"fit_cast_condition: no activations captured at layer {li}.")
        result[li] = torch.stack(out[li], dim=0)
    return result


def fit_cast_condition(
    model: nn.Module,
    harmful_prompts: List[str],
    benign_prompts: List[str],
    tokenizer: Any,
    candidate_layers: Optional[List[int]] = None,
    *,
    pool: str = "last",
    threshold_range: Tuple[float, float] = (-1.0, 1.0),
    threshold_step: float = 0.01,
    val_frac: float = 0.5,
    device: str = "cpu",
) -> CASTCondition:
    """Fit a faithful CAST condition (cosine-similarity gate) via grid search.

    Faithful to ``MalleableModel.find_best_condition_point`` in
    IBM/activation-steering:

      1. Pool hidden states (``pool``) at each candidate condition layer for
         condition-present (``harmful_prompts``) and condition-absent
         (``benign_prompts``) prompts.
      2. Form the **condition vector** = L2-normalized difference-of-means
         (harmful mean − benign mean) at each layer (IBM uses PCA-1 of the same
         contrast; the leading component is the diff-of-means direction).
      3. Compute the cosine condition similarity (``_cast_similarity``:
         ``cos(h, tanh(P·h))`` with ``P = vᵀv / (v·v)``) for every prompt.
      4. **Grid-search** ``(layer, threshold, comparator)`` over
         ``threshold ∈ arange(*threshold_range, threshold_step)`` and
         ``comparator ∈ {"larger", "smaller"}``, selecting the point that
         **maximizes the condition F1** (fire on harmful, not on benign). F1 is
         scored on a held-out split (so the reported number is not overfit); ties
         are broken by the threshold's *margin* to the nearest similarity and by
         the class-mean separation, so layers with genuine geometric separation
         are preferred over a lucky threshold on a near-degenerate layer.

    The condition vector is fit on the training split; the threshold/comparator
    are selected by held-out F1.

    Args:
        candidate_layers: layers to grid-search. ``None`` → all decoder layers.
        pool: ``"mean"`` (paper default) or ``"last"`` token pooling.
        threshold_range / threshold_step: cosine-similarity threshold grid.
        val_frac: fraction of each class held out for threshold selection.

    Returns:
        :class:`CASTCondition` with the chosen ``(condition_vector,
        condition_layer, threshold, comparator, f1, pool)``.
    """
    layers = _get_decoder_layers(model)
    if candidate_layers is None:
        # CAST checks the condition at a mid-to-late layer (the paper grid-
        # searches a layer range, not the very first embedding-like blocks).
        n = len(layers)
        lo, hi = max(1, n // 4), max(2, (3 * n) // 4 + 1)
        candidate_layers = list(range(lo, min(hi, n)))

    H = _collect_pooled_hidden(model, tokenizer, harmful_prompts, candidate_layers, pool)
    B = _collect_pooled_hidden(model, tokenizer, benign_prompts, candidate_layers, pool)

    def _split(n: int) -> Tuple[List[int], List[int]]:
        n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
        # Deterministic interleaved split so both halves cover the set.
        val_idx = list(range(0, n, max(1, n // max(1, n_val))))[:n_val]
        val_set = set(val_idx)
        tr_idx = [i for i in range(n) if i not in val_set]
        if not tr_idx:  # tiny set: reuse all for train
            tr_idx = list(range(n))
        if not val_idx:  # tiny set: validate on train
            val_idx = list(range(n))
        return tr_idx, val_idx

    nh, nb = H[candidate_layers[0]].shape[0], B[candidate_layers[0]].shape[0]
    h_tr, h_val = _split(nh)
    b_tr, b_val = _split(nb)

    import numpy as np

    thresholds = np.arange(threshold_range[0], threshold_range[1] + 1e-9, threshold_step)
    comparators = ["larger", "smaller"]

    best: Optional[CASTCondition] = None
    # key = (held_out_f1, margin, class_separation) — all higher is better.
    best_key: Tuple[float, float, float] = (-1.0, -1e9, -1e9)

    for li in candidate_layers:
        Hh, Bb = H[li], B[li]
        # Condition vector fit on TRAIN split only (held-out F1 below).
        cv = Hh[h_tr].mean(dim=0) - Bb[b_tr].mean(dim=0)
        nrm = cv.norm()
        if nrm < 1e-9:
            continue
        cv = cv / nrm

        sims_h = np.array([_cast_similarity(Hh[i], cv) for i in range(nh)])
        sims_b = np.array([_cast_similarity(Bb[i], cv) for i in range(nb)])

        # Class-mean separation (signed): positive when harmful sims are higher.
        sep = float(sims_h.mean() - sims_b.mean())

        vh = sims_h[h_val]
        vb = sims_b[b_val]
        y_true = [1] * len(vh) + [0] * len(vb)
        sims_val = np.concatenate([vh, vb])

        for thr in thresholds:
            for comp in comparators:
                y_pred = [1 if _condition_met(float(s), float(thr), comp) else 0 for s in sims_val]
                f1 = _f1(y_true, y_pred)
                if f1 <= 0:
                    continue
                # Margin: how far the threshold sits from the nearest val point.
                # Larger margin → more robust decision boundary.
                margin = float(np.min(np.abs(sims_val - thr)))
                key = (f1, margin, abs(sep))
                if key > best_key:
                    best_key = key
                    best = CASTCondition(
                        condition_vector=cv.clone(),
                        condition_layer=int(li),
                        threshold=float(thr),
                        comparator=comp,
                        f1=float(f1),
                        pool=pool,
                    )

    if best is None:
        # No separating point found — fall back to mid layer, mean-sim midpoint.
        li = candidate_layers[len(candidate_layers) // 2]
        Hh, Bb = H[li], B[li]
        cv = Hh.mean(dim=0) - Bb.mean(dim=0)
        cv = cv / cv.norm().clamp_min(1e-9)
        sims_h = [_cast_similarity(Hh[i], cv) for i in range(nh)]
        sims_b = [_cast_similarity(Bb[i], cv) for i in range(nb)]
        mid = 0.5 * (float(np.mean(sims_h)) + float(np.mean(sims_b)))
        best = CASTCondition(
            condition_vector=cv, condition_layer=int(li),
            threshold=mid, comparator="smaller", f1=0.0, pool=pool,
        )
        logger.warning("fit_cast_condition: no separating point found; using fallback.")

    logger.info(
        "fit_cast_condition: layer=%d threshold=%.3f comparator=%s F1=%.3f "
        "(harmful=%d benign=%d pool=%s)",
        best.condition_layer, best.threshold, best.comparator, best.f1, nh, nb, pool,
    )
    return best


def _f1(y_true: List[int], y_pred: List[int]) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# Back-compat: logistic-probe-style fitting (DEPRECATED).
# ---------------------------------------------------------------------------

def fit_cast_probe(
    model: nn.Module,
    harmful_prompts: List[str],
    benign_prompts: List[str],
    tokenizer: Any,
    probe_layer: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, float]:
    """DEPRECATED back-compat shim.

    The original SafeTune CAST variant gated on a logistic probe
    ``sigmoid(wᵀh + b) > threshold``, which is *not* the paper's mechanism.
    CAST is now implemented with a cosine-similarity condition vector
    (:func:`fit_cast_condition` + :class:`CASTModel`). This shim is retained so
    legacy callers that expect ``(weight, bias)`` keep working; it returns the
    normalized difference-of-means direction at ``probe_layer`` and a midpoint
    bias. Prefer :func:`fit_cast_condition`.
    """
    logger.warning(
        "fit_cast_probe is deprecated; CAST now uses a cosine-similarity "
        "condition vector. Use fit_cast_condition instead."
    )
    pooled = _collect_pooled_hidden(model, tokenizer, harmful_prompts, [probe_layer], pool="last")
    Xh = pooled[probe_layer]
    pooled_b = _collect_pooled_hidden(model, tokenizer, benign_prompts, [probe_layer], pool="last")
    Xb = pooled_b[probe_layer]

    w = Xh.mean(dim=0) - Xb.mean(dim=0)
    w = w / w.norm().clamp_min(1e-12)
    h_proj = (Xh * w).sum(dim=-1)
    b_proj = (Xb * w).sum(dim=-1)
    bias = float(-0.5 * (h_proj.mean() + b_proj.mean()))
    return w, bias


# ---------------------------------------------------------------------------
# Runtime model wrapper
# ---------------------------------------------------------------------------

class CASTModel:
    """Conditional Activation Steering — adds the CAA behavior vector only when
    a **cosine-similarity condition gate** fires (arXiv:2409.05907).

    Faithful to IBM/activation-steering: at the **prefill (prompt) pass** the
    pooled hidden state at ``condition_layer`` is compared to the
    ``condition_vector`` by cosine similarity (``_cast_similarity``); if the
    learned comparator/threshold says the condition is met, the CAA steering
    vectors are installed and the forward pass is re-run; otherwise the input
    passes through unmodified.

    Construction (recommended)::

        cond = fit_cast_condition(model, harmful, benign, tok)
        vecs = extract_caa_vectors(model, tok, harmful, benign, cfg)
        cast = CASTModel(model, vecs, condition=cond, alpha=1.0)
        with cast:
            out = model.generate(**inputs)

    Back-compat construction (legacy logistic probe) is still accepted via the
    ``probe_layer`` / ``probe_weights`` / ``probe_bias`` arguments; in that mode
    the gate falls back to ``sigmoid(wᵀh + b) > threshold``.
    """

    def __init__(
        self,
        model: nn.Module,
        steering_vectors: Dict[int, torch.Tensor],
        condition: Optional[CASTCondition] = None,
        *,
        alpha: float = 1.0,
        # --- back-compat (legacy logistic probe) ---
        probe_layer: Optional[int] = None,
        probe_weights: Optional[torch.Tensor] = None,
        probe_bias: float = 0.0,
        threshold: Optional[float] = None,
    ) -> None:
        self.model = model
        self.steering_vectors = {int(k): v.detach().clone() for k, v in steering_vectors.items()}
        self.alpha = float(alpha)
        self._handles: List[Any] = []

        self.condition = condition
        self._legacy_probe = condition is None and probe_weights is not None

        if self.condition is not None:
            self.condition_layer = int(condition.condition_layer)
            self.condition_vector = condition.condition_vector.detach().clone()
            self.threshold = float(condition.threshold)
            self.comparator = condition.comparator
            self.pool = condition.pool
        elif self._legacy_probe:
            # Legacy logistic-probe gate.
            self.condition_layer = int(probe_layer)
            self.probe_weights = probe_weights.detach().clone()
            self.probe_bias = float(probe_bias)
            self.threshold = float(threshold if threshold is not None else 0.5)
            self.pool = "last"
        else:
            raise ValueError(
                "CASTModel requires either a `condition` (CASTCondition) or the "
                "legacy `probe_weights`/`probe_layer` arguments."
            )

    # ------------------------------------------------------------------
    # Condition evaluation (prefill pass)
    # ------------------------------------------------------------------

    def _gate_fires(self, input_ids: torch.Tensor, **kwargs: Any) -> Tuple[bool, float]:
        """Run a condition-only prefill pass and decide whether the gate fires.

        Returns ``(fires, similarity_or_score)``.
        """
        layers = _get_decoder_layers(self.model)
        if not (0 <= self.condition_layer < len(layers)):
            raise IndexError(
                f"CASTModel: condition_layer {self.condition_layer} out of range "
                f"(model has {len(layers)} decoder layers)."
            )

        captured: List[torch.Tensor] = []

        def _hook(_m: nn.Module, _i: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            captured.append(h[0].detach().float().cpu())  # (seq, hidden) for seq 0

        handle = layers[self.condition_layer].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                self.model(input_ids=input_ids, **kwargs)
        finally:
            handle.remove()

        if not captured:
            logger.warning("CASTModel: condition hook did not fire; gate defaults to OFF.")
            return False, 0.0

        h_seq = captured[-1]

        if self._legacy_probe:
            h = _pool_hidden(h_seq, self.pool)
            w = self.probe_weights.to(h.device)
            logit = float((h * w).sum()) + self.probe_bias
            score = float(torch.sigmoid(torch.tensor(logit)))
            return (score > self.threshold), score

        h = _pool_hidden(h_seq, self.pool)
        sim = _cast_similarity(h, self.condition_vector)
        return _condition_met(sim, self.threshold, self.comparator), sim

    # ------------------------------------------------------------------
    # Steering hooks (behavior vector)
    # ------------------------------------------------------------------

    def _make_steering_hook(self, vec: torch.Tensor) -> Any:
        def hook(_m: nn.Module, _i: Any, out: Any) -> Any:
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            v = vec.to(dtype=h.dtype, device=h.device)
            h = h + self.alpha * v
            return (h,) + out[1:] if is_tuple else h
        return hook

    def _install_steering(self) -> None:
        self._remove_steering()
        layers = _get_decoder_layers(self.model)
        for idx, vec in self.steering_vectors.items():
            if 0 <= idx < len(layers):
                self._handles.append(
                    layers[idx].register_forward_hook(self._make_steering_hook(vec))
                )
        logger.debug(
            "CASTModel: installed %d behavior hooks (alpha=%.2f).",
            len(self._handles), self.alpha,
        )

    def _remove_steering(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        """Conditionally-steered forward pass (arXiv:2409.05907).

        1. Prefill condition pass at ``condition_layer``; compute the cosine
           condition similarity ``sim``.
        2. If the comparator/threshold says the condition is met: install the
           CAA behavior hooks, re-run the forward pass, remove the hooks.
        3. Otherwise: return the unmodified forward pass.
        """
        fires, score = self._gate_fires(input_ids, **kwargs)
        logger.debug("CASTModel: gate=%s (score/sim=%.4f thr=%.4f).", fires, score, self.threshold)
        if fires:
            self._install_steering()
            try:
                return self.model(input_ids=input_ids, **kwargs)
            finally:
                self._remove_steering()
        return self.model(input_ids=input_ids, **kwargs)

    def install(self) -> "CASTModel":
        """Install permanent behavior hooks (use when calling model.generate directly).

        Note: this installs *unconditionally*. For the conditional gate, route
        generation through :meth:`forward` / :meth:`generate` instead.
        """
        self._install_steering()
        return self

    def remove(self) -> None:
        self._remove_steering()

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Conditional generation: gate on the prompt, then steer if it fires.

        Mirrors ``MalleableModel.respond`` — the condition is decided once, at
        prefill, from the prompt; behavior steering is then applied (or not) for
        the whole generation.
        """
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        attn = kwargs.get("attention_mask")
        fires, score = (False, 0.0)
        if input_ids is not None:
            gate_kwargs = {"attention_mask": attn} if attn is not None else {}
            fires, score = self._gate_fires(input_ids, **gate_kwargs)
        logger.debug("CASTModel.generate: gate=%s (sim=%.4f thr=%.4f).", fires, score, self.threshold)
        if fires:
            self._install_steering()
            try:
                return self.model.generate(*args, **kwargs)
            finally:
                self._remove_steering()
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        return getattr(self.__dict__["model"], name)

    def __enter__(self) -> "CASTModel":
        # Context-manager installs permanent (unconditional) hooks for callers
        # that drive model.generate directly. Use .generate()/.forward() for the
        # conditional gate.
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.remove()


__all__ = ["CASTModel", "CASTCondition", "fit_cast_condition", "fit_cast_probe"]
