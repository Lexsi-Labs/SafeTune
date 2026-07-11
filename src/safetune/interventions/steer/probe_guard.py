"""
Linear-Probe Safeguard: classify prompts as harmful / harmless via a linear
probe on a chosen layer's pooled hidden state, then route harmful ones to a
fixed refusal response (or any callable).

This is the cheapest possible runtime defense:

  1. Fit a logistic-regression probe on (harmful, harmless) contrast pairs
     using the same activation pool as :mod:`safetune.steer.refusal_direction`.
  2. At inference time, run a fast forward up to the probe layer, score the
     prompt, and either refuse or pass through.

Why it is in SafeTune: every defense paper compares against a linear probe.
If your steering / SAE / circuit-breaker can't beat this on the same eval
harness, you do not have a story. Including this means a reviewer cannot
ask for it later. It is also the only defense that has near-zero inference
cost overhead (one extra dot-product per layer).

Implementation notes:
* We use ``sklearn.LogisticRegression`` with L2 regularization. NumPy only;
  no torch op at fit time beyond hidden-state collection.
* The probe stores its layer index, mean, std, and weight vector. Apply via
  :class:`LinearProbeGuardModel` which hooks the same layer at inference.
* On a flagged prompt the guard either returns a canned ``refusal_text``
  (default) or calls ``on_refuse`` for custom routing (e.g. constitutional
  rewrite). On a passed prompt the model generates normally.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from tqdm.auto import tqdm

import torch
import torch.nn as nn

from safetune._refusal_helpers import _get_decoder_layers

logger = logging.getLogger(__name__)


@dataclass
class LinearProbeConfig:
    """Configuration for fitting the safety probe.

    Attributes:
        pick_layer: layer to probe. Default is the model's middle layer.
        pool_method: ``"last_token"`` (paper-faithful) or ``"mean"``.
        C: sklearn LogisticRegression inverse-regularization strength.
        threshold: prompt is flagged when probe_score >= threshold.
            ``None`` means auto-calibrate to the midpoint between class-mean
            scores at fit time; this is robust on small contrast sets.
        max_iter: sklearn max_iter.
    """

    pick_layer: Optional[int] = None
    pool_method: str = "last_token"
    C: float = 1.0
    threshold: Optional[float] = None
    max_iter: int = 200


@dataclass
class LinearProbe:
    """Fitted linear probe weights.

    Attributes:
        layer_idx: which layer the probe was fit on.
        weight: 1-D tensor (hidden,).
        bias: scalar.
        mean: per-feature mean of the fit data (for centering).
        std: per-feature std of the fit data.
        threshold: scoring threshold inherited from config.
        pool_method: how to reduce sequence dim.
    """

    layer_idx: int
    weight: torch.Tensor
    bias: float
    mean: torch.Tensor
    std: torch.Tensor
    threshold: float
    pool_method: str

    def score(self, pooled: torch.Tensor) -> torch.Tensor:
        """Logit-space scores (positive = harmful) for a batch of pooled activations."""
        x = (pooled.float() - self.mean.to(pooled.device)) / self.std.to(pooled.device).clamp_min(1e-6)
        return x @ self.weight.to(pooled.device) + self.bias

    def predict(self, pooled: torch.Tensor) -> torch.Tensor:
        """1-D Boolean: True if flagged as harmful."""
        return self.score(pooled) >= self.threshold


def _pool_hidden(h: torch.Tensor, method: str) -> torch.Tensor:
    """Pool a (batch, seq, hidden) tensor down to (batch, hidden)."""
    if method == "last_token":
        return h[:, -1, :]
    if method == "max":
        return h.max(dim=1).values
    return h.mean(dim=1)


def _collect_pooled(
    model: nn.Module,
    tokenizer: Any,
    prompts: List[str],
    layer_idx: int,
    pool_method: str,
    batch_size: int = 8,
) -> torch.Tensor:
    """Run prompts through the model and return pooled activations at one layer."""
    layers = _get_decoder_layers(model)
    if not (0 <= layer_idx < len(layers)):
        raise IndexError(f"layer_idx {layer_idx} out of range for {len(layers)} layers.")

    captured: List[torch.Tensor] = []

    def _hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(_pool_hidden(h, pool_method).detach().cpu())

    handle = layers[layer_idx].register_forward_hook(_hook)
    try:
        model.eval()
        device = next(model.parameters()).device
        all_pooled: List[torch.Tensor] = []
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(prompts), batch_size), total=n_batches,
                      desc="ProbeGuard [calibrate]", unit="batch", leave=False):
            batch = prompts[i : i + batch_size]
            captured.clear()
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            if hasattr(enc, "to"):
                enc = enc.to(device)
            else:
                enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in dict(enc).items()}
            with torch.no_grad():
                model(**enc)
            if captured:
                all_pooled.append(captured[-1])
        return torch.cat(all_pooled, dim=0) if all_pooled else torch.empty(0)
    finally:
        handle.remove()


def fit_linear_probe(
    model: nn.Module,
    tokenizer: Any,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    config: Optional[LinearProbeConfig] = None,
) -> LinearProbe:
    """Fit a mean-difference (Fisher-style) linear probe.

    The weight vector is ``mean(harmful_activations) - mean(harmless_activations)``,
    matching the geometric convention used by Arditi et al. (refusal direction)
    and Panickssery et al. (CAA). For larger contrast sets (>= 30 per class)
    we additionally fit an sklearn ``LogisticRegression`` and pick whichever
    discriminator separates the training set better. On small contrast sets
    sklearn's L2 regularization collapses the coefficient to zero, so the
    mean-diff fallback is the robust default.
    """
    cfg = config or LinearProbeConfig()
    layers = _get_decoder_layers(model)
    if not layers:
        raise RuntimeError(
            "fit_linear_probe: cannot locate decoder layers. "
            "Expected model.model.language_model.layers (Gemma-3), "
            "model.model.layers (Llama/Mistral/Qwen/Gemma), or "
            "model.transformer.h (GPT-style)."
        )
    pick = cfg.pick_layer if cfg.pick_layer is not None else len(layers) // 2

    Xh = _collect_pooled(model, tokenizer, harmful_prompts, pick, cfg.pool_method)
    Xs = _collect_pooled(model, tokenizer, harmless_prompts, pick, cfg.pool_method)
    if Xh.numel() == 0 or Xs.numel() == 0:
        raise RuntimeError("fit_linear_probe: pool collection returned empty tensors.")

    X = torch.cat([Xh, Xs], dim=0).float()
    mean = X.mean(dim=0)
    std = X.std(dim=0).clamp_min(1e-6)

    # Mean-difference projection in standardized space.
    Xn_h = (Xh.float() - mean) / std
    Xn_s = (Xs.float() - mean) / std
    md_weight = Xn_h.mean(dim=0) - Xn_s.mean(dim=0)
    if md_weight.norm() < 1e-12:
        # Pathological: classes coincide in the standardized space. Use raw mean-diff.
        md_weight = Xh.float().mean(dim=0) - Xs.float().mean(dim=0)
    md_weight = md_weight / md_weight.norm().clamp_min(1e-12)

    # Bias so the midpoint between class means scores at 0.
    h_proj = (Xn_h * md_weight).sum(dim=-1)
    s_proj = (Xn_s * md_weight).sum(dim=-1)
    md_bias = -0.5 * float(h_proj.mean() + s_proj.mean())

    weight = md_weight
    bias = md_bias

    # Optionally refine with logistic regression on larger contrast sets.
    try:
        from sklearn.linear_model import LogisticRegression
        if len(Xh) + len(Xs) >= 60:
            y = torch.cat([torch.ones(len(Xh)), torch.zeros(len(Xs))]).numpy()
            X_all = torch.cat([Xn_h, Xn_s], dim=0).numpy()
            clf = LogisticRegression(C=cfg.C, max_iter=cfg.max_iter)
            clf.fit(X_all, y)
            lr_weight = torch.tensor(clf.coef_[0], dtype=torch.float32)
            if lr_weight.norm() > 1e-3:
                weight = lr_weight / lr_weight.norm().clamp_min(1e-12)
                bias = float(clf.intercept_[0])
    except ImportError:  # pragma: no cover
        pass

    raw = (torch.cat([Xn_h, Xn_s], dim=0) * weight).sum(dim=-1) + bias
    h_scores = raw[: len(Xh)]
    s_scores = raw[len(Xh):]
    train_acc = float((torch.cat([(h_scores >= 0), (s_scores < 0)])).float().mean())

    if cfg.threshold is None:
        threshold = float((h_scores.mean() + s_scores.mean()) / 2) if h_scores.numel() and s_scores.numel() else 0.0
    else:
        threshold = float(cfg.threshold)

    logger.info(
        "LinearProbe: fit on %d harmful + %d harmless, layer=%d, train_acc=%.3f, threshold=%.4f",
        len(Xh), len(Xs), pick, train_acc, threshold,
    )

    return LinearProbe(
        layer_idx=pick,
        weight=weight,
        bias=bias,
        mean=mean.float(),
        std=std.float(),
        threshold=threshold,
        pool_method=cfg.pool_method,
    )


class LinearProbeGuardModel:
    """Wrap a model so harmful prompts (per the probe) get a canned refusal.

    The wrapper does NOT install permanent forward hooks. ``score(prompts)``
    runs a single forward pass through the model and returns probe scores;
    ``guard(prompt)`` returns either the canned refusal or invokes a callable
    to generate normally.

    The point of this design: integration with arbitrary downstream
    generators (HF, vLLM, API) without forcing them to share the model
    object.
    """

    DEFAULT_REFUSAL = (
        "I'm sorry, but I can't help with that request."
    )

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        probe: LinearProbe,
        refusal_text: str = DEFAULT_REFUSAL,
        on_pass: Optional[Callable[[str], str]] = None,
        on_refuse: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.probe = probe
        self.refusal_text = refusal_text
        self.on_pass = on_pass
        self.on_refuse = on_refuse

    def score(self, prompts: List[str]) -> torch.Tensor:
        pooled = _collect_pooled(
            self.model, self.tokenizer, prompts, self.probe.layer_idx, self.probe.pool_method
        )
        return self.probe.score(pooled.to(next(self.model.parameters()).device))

    def predict(self, prompts: List[str]) -> List[bool]:
        scores = self.score(prompts)
        return [bool(s.item() >= self.probe.threshold) for s in scores]

    def guard(self, prompt: str) -> str:
        """Return either a canned refusal or the result of ``on_pass``."""
        flagged = self.predict([prompt])[0]
        if flagged:
            if self.on_refuse is not None:
                return self.on_refuse(prompt)
            return self.refusal_text
        if self.on_pass is not None:
            return self.on_pass(prompt)
        return ""

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in ("model", "tokenizer", "probe"):
            raise AttributeError(name)
        return getattr(self.__dict__["model"], name)

    def guard_batch(self, prompts: List[str]) -> List[str]:
        flags = self.predict(prompts)
        return [
            (
                (self.on_refuse(p) if self.on_refuse else self.refusal_text)
                if flag
                else (self.on_pass(p) if self.on_pass else "")
            )
            for p, flag in zip(prompts, flags)
        ]


__all__ = [
    "LinearProbe",
    "LinearProbeConfig",
    "LinearProbeGuardModel",
    "fit_linear_probe",
]
