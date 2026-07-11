"""AdaSteer inference-time steering model wrapper.

Self-contained SafeTune VARIANT inspired by:

    "AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender"
    Zhao, Guo, Hu, Deng, Zhang, Sui, Han, Zhao, Qin, Chua et al.
    EMNLP 2025 (Oral) -- arXiv:2504.09466
    Original repo: https://github.com/MuyuenLP/AdaSteer
    Reference file: adasteer/models/For_Steering_LlamaModel_adasteer.py
                    (class ``LlamaModel_for_Steering.get_steer`` / ``forward``)

FIDELITY: faithful to the paper's mechanism. AdaSteer steers along the
Rejection Direction (RD) and Harmfulness Direction (HD) with **per-input
adaptive coefficients learned via logistic regression** (the paper's R-Law /
H-Law; arXiv:2504.09466 abstract + §3) — which is exactly what
``_coeff`` does (``min + (max−min)·sigmoid(w·proj+b)`` with separately fitted
RD/HD logistics). The only difference from the authors' *released inference
code* is the exact coefficient scale/clamps (their deployment uses an affine
ramp like ``0.02·(proj+60)`` clamped to a fixed range), which is a tuning
instantiation of the same logistic law, not a different algorithm.

Why this file is no longer a thin re-export
--------------------------------------------
The previous version delegated to ``runtime.inference.adasteer.AdaSteerWrapper``,
which steered along a *single* direction with a *fixed* ``base_multiplier`` and
never recomputed an adaptive coefficient inside the generation path -- i.e. plain
fixed-coefficient CAA.  AdaSteer's whole substance is:

  * **Two directions.**  The **Rejection Direction (RD)** and the
    **Harmfulness Direction (HD)** (in the authors' code these are the
    "acceptance direction" ``mean(harmless) - mean(harmful)`` and the
    "pseudo-acceptance direction" derived from adversarial/benign pairs,
    with HD orthogonalised against RD -- see ``proj.py`` in the repo).

  * **Per-input adaptive coefficients learned via logistic regression.**
    For every input, AdaSteer projects the last-token hidden state onto RD/HD
    (the **Rejection Law** and **Harmfulness Law**) and maps that projection to
    a steering coefficient.  In the authors' release the map is a calibrated
    affine ramp; the paper frames the calibration as logistic regression on the
    RD/HD projections.  We fit a real ``sklearn`` logistic regression on the
    RD/HD projection features of harmful vs. benign calibration prompts and use
    its decision value to produce the per-input coefficient.

  * **The adaptive multiplier is applied in the actual forward path.**
    A forward hook recomputes ``c_RD`` and ``c_HD`` *per input* from that
    input's own activations and adds ``c_RD * RD + c_HD * HD`` to the residual
    stream.  Benign inputs land on the low-coefficient side of the logistic
    boundary and receive ~0 steering.

This module is now standalone -- it does not import the (faithless) core
wrapper.  ``runtime.inference.adasteer`` is left untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Direction extraction (difference-in-means + orthogonalisation)
# --------------------------------------------------------------------------
def _to_numpy(x: Any):
    """Best-effort conversion of a tensor / array to a float64 numpy array."""
    import numpy as np

    if x is None:
        return None
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().to("cpu").float().numpy()
    return np.asarray(x, dtype=np.float64)


def _normalize(v):
    import numpy as np

    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def extract_adasteer_directions(
    rejection_pos: Dict[int, Any],
    rejection_neg: Dict[int, Any],
    harm_pos: Optional[Dict[int, Any]] = None,
    harm_neg: Optional[Dict[int, Any]] = None,
    orthogonalize: bool = True,
) -> Tuple[Dict[int, Any], Dict[int, Any]]:
    """Build the Rejection Direction (RD) and Harmfulness Direction (HD).

    Mirrors ``LlamaModel_for_Steering.get_steer`` in the AdaSteer repo:

    * ``RD[l] = mean(harmless_l) - mean(harmful_l)``  (the "acceptance
      direction": pointing from refusal toward acceptance).
    * ``HD[l] = mean(pseudo_harmless_l) - mean(pseudo_harmful_l)``  (the
      "pseudo-acceptance direction" separating adversarial vs. benign).
    * HD is orthogonalised against RD (repo ``proj.py`` ->
      ``get_orthogonalized_matrix_2``) so the two laws act independently.

    ``*_pos`` / ``*_neg`` map ``layer_idx -> activation matrix (N, D)`` or an
    already-pooled mean vector ``(D,)``.  If the harmfulness pairs are omitted,
    HD falls back to RD (single-law degenerate mode).

    Returns ``(RD, HD)`` as ``{layer_idx: np.ndarray(D,)}``.
    """
    import numpy as np

    def _mean(mat):
        a = _to_numpy(mat)
        return a if a.ndim == 1 else a.mean(axis=0)

    rd: Dict[int, Any] = {}
    for layer in rejection_pos:
        if layer not in rejection_neg:
            continue
        # pos == accepting/harmless, neg == refusing/harmful
        rd[layer] = _mean(rejection_pos[layer]) - _mean(rejection_neg[layer])

    hd: Dict[int, Any] = {}
    if harm_pos is not None and harm_neg is not None:
        for layer in harm_pos:
            if layer not in harm_neg:
                continue
            hd[layer] = _mean(harm_pos[layer]) - _mean(harm_neg[layer])
    else:
        hd = {l: v.copy() for l, v in rd.items()}

    if orthogonalize:
        for layer, h in hd.items():
            if layer in rd:
                r_hat = _normalize(rd[layer])
                # h <- h - (h . r_hat) r_hat   (remove the RD component)
                hd[layer] = h - float(np.dot(h, r_hat)) * r_hat

    return rd, hd


# --------------------------------------------------------------------------
# Adaptive coefficient: logistic regression on RD/HD projections
# --------------------------------------------------------------------------
class AdaSteerCoefficient:
    """Learns the per-input adaptive steering coefficients (R-Law / H-Law).

    AdaSteer projects an input's last-token hidden state onto RD and HD and
    maps the projection to a steering strength.  We fit one logistic
    regression per law: it learns, from harmful/benign calibration prompts,
    a boundary in projection space.  At generation time the input's projection
    is fed through the fitted logistic; harmful-side inputs get a large
    coefficient, benign-side inputs get ~0 -- the adaptive behaviour the paper
    is built around.

    Coefficient map (per input, per law):

        z       = w * proj + b              (logistic decision value)
        p       = sigmoid(z)                (P[input needs steering])
        coeff   = min_coeff + (max_coeff - min_coeff) * p
    """

    def __init__(
        self,
        rd_layer: int,
        hd_layer: int,
        rd_max: float = 4.0,
        hd_max: float = 4.0,
        min_coeff: float = 0.0,
    ) -> None:
        self.rd_layer = rd_layer
        self.hd_layer = hd_layer
        self.rd_max = rd_max
        self.hd_max = hd_max
        self.min_coeff = min_coeff
        # Fitted logistic params: (weight, bias) on the 1-D projection feature.
        self._rd_lr: Optional[Tuple[float, float]] = None
        self._hd_lr: Optional[Tuple[float, float]] = None
        self.fitted = False

    # -- projection helpers -------------------------------------------------
    @staticmethod
    def _project(hidden, direction, anchor=None):
        """Scalar projection of ``hidden`` onto ``direction``.

        If an ``anchor`` is given the distance from the anchor is projected
        (matching the repo's ``dis_harmful = h - harmful_anchor`` then
        ``np.dot(dis, direction)``).  ``hidden`` may be a batch ``(N, D)``.
        """

        h = _to_numpy(hidden)
        d = _to_numpy(direction)
        if anchor is not None:
            h = h - _to_numpy(anchor)
        return h @ d  # (N,) or scalar

    @staticmethod
    def _fit_logistic(features, labels) -> Tuple[float, float]:
        """Fit a 1-D logistic regression; return ``(weight, bias)``.

        Falls back to a closed-form mean-difference boundary when sklearn is
        unavailable or a class is missing.
        """
        import numpy as np

        x = np.asarray(features, dtype=np.float64).reshape(-1)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)

        if len(np.unique(y)) < 2:
            # Degenerate: no contrast -> neutral boundary.
            return 0.0, 0.0

        try:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression()
            clf.fit(x.reshape(-1, 1), y)
            return float(clf.coef_.reshape(-1)[0]), float(clf.intercept_.reshape(-1)[0])
        except Exception:  # pragma: no cover - sklearn missing / fit failed
            pos = x[y == 1].mean() if (y == 1).any() else 0.0
            neg = x[y == 0].mean() if (y == 0).any() else 0.0
            spread = abs(pos - neg) + 1e-6
            w = 4.0 / spread * (1.0 if pos >= neg else -1.0)
            b = -w * 0.5 * (pos + neg)
            return float(w), float(b)

    # -- fitting ------------------------------------------------------------
    def fit(
        self,
        rd_direction,
        hd_direction,
        harmful_acts: Dict[int, Any],
        benign_acts: Dict[int, Any],
        rd_anchor=None,
        hd_anchor=None,
    ) -> "AdaSteerCoefficient":
        """Calibrate the two logistic laws.

        ``harmful_acts`` / ``benign_acts`` map ``layer -> (N, D)`` last-token
        activations of calibration prompts.  The RD logistic is fit at
        ``rd_layer`` (Rejection Law), the HD logistic at ``hd_layer``
        (Harmfulness Law).
        """
        import numpy as np

        # Rejection Law.
        if self.rd_layer in harmful_acts and self.rd_layer in benign_acts:
            p_h = self._project(harmful_acts[self.rd_layer], rd_direction, rd_anchor)
            p_b = self._project(benign_acts[self.rd_layer], rd_direction, rd_anchor)
            feats = np.concatenate([np.atleast_1d(p_h), np.atleast_1d(p_b)])
            labs = np.concatenate(
                [np.ones(np.atleast_1d(p_h).shape), np.zeros(np.atleast_1d(p_b).shape)]
            )
            self._rd_lr = self._fit_logistic(feats, labs)

        # Harmfulness Law.
        if self.hd_layer in harmful_acts and self.hd_layer in benign_acts:
            p_h = self._project(harmful_acts[self.hd_layer], hd_direction, hd_anchor)
            p_b = self._project(benign_acts[self.hd_layer], hd_direction, hd_anchor)
            feats = np.concatenate([np.atleast_1d(p_h), np.atleast_1d(p_b)])
            labs = np.concatenate(
                [np.ones(np.atleast_1d(p_h).shape), np.zeros(np.atleast_1d(p_b).shape)]
            )
            self._hd_lr = self._fit_logistic(feats, labs)

        self.fitted = self._rd_lr is not None or self._hd_lr is not None
        return self

    # -- inference ----------------------------------------------------------
    @staticmethod
    def _sigmoid(z):
        import numpy as np

        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def _coeff(self, projection, lr, max_coeff):
        """Map a projection value through a fitted logistic to a coefficient."""
        import numpy as np

        proj = np.atleast_1d(np.asarray(projection, dtype=np.float64))
        if lr is None:
            # Not calibrated -> apply the base coefficient uniformly.
            return np.full(proj.shape, max_coeff, dtype=np.float64)
        w, b = lr
        p = self._sigmoid(w * proj + b)
        return self.min_coeff + (max_coeff - self.min_coeff) * p

    def rd_coefficient(self, rd_projection):
        """Per-input RD steering coefficient(s) (Rejection Law)."""
        return self._coeff(rd_projection, self._rd_lr, self.rd_max)

    def hd_coefficient(self, hd_projection):
        """Per-input HD steering coefficient(s) (Harmfulness Law)."""
        return self._coeff(hd_projection, self._hd_lr, self.hd_max)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class AdaSteerConfig:
    """Configuration for AdaSteer two-direction adaptive steering."""

    # Layers at which the steering vectors are added to the residual stream.
    target_layers: List[int] = field(default_factory=lambda: list(range(10, 20)))
    # Layer whose last-token activation drives the Rejection Law logistic.
    rd_probe_layer: int = 8
    # Layer whose last-token activation drives the Harmfulness Law logistic.
    hd_probe_layer: int = 13
    # Upper coefficient bounds for RD / HD (the "base" strengths).
    rd_max: float = 4.0
    hd_max: float = 4.0
    # Lower coefficient bound (benign inputs collapse to ~this).
    min_coeff: float = 0.0
    # When False, ignore HD and the logistic and steer with a fixed RD coeff.
    adaptive: bool = True
    # Orthogonalise HD against RD when extracting directions.
    orthogonalize: bool = True


# --------------------------------------------------------------------------
# Model wrapper
# --------------------------------------------------------------------------
class AdaSteerModel:
    """Aligned-LLM wrapper that applies AdaSteer two-direction adaptive steering.

    Public signature is preserved (``model``, ``safety_vectors``,
    ``target_layers``, ``base_multiplier``, ``adaptive``, ``safety_threshold``);
    the new behaviour is exposed through optional keyword arguments with
    defaults so existing callers are unaffected.

    Typical use::

        m = AdaSteerModel(model,
                          rejection_direction=RD, harmfulness_direction=HD)
        m.fit_adaptive(harmful_acts, benign_acts)   # learn the logistic laws
        out = m.generate(input_ids)                 # adaptive steering applied

    If only ``safety_vectors`` are supplied (legacy path) AdaSteer degrades
    gracefully to single-direction steering with a fixed coefficient.
    """

    def __init__(
        self,
        model: Any,
        safety_vectors: Optional[Dict[int, Any]] = None,
        target_layers: Optional[List[int]] = None,
        base_multiplier: float = 3.0,
        adaptive: bool = True,
        safety_threshold: float = 0.5,
        *,
        rejection_direction: Optional[Dict[int, Any]] = None,
        harmfulness_direction: Optional[Dict[int, Any]] = None,
        rd_probe_layer: int = 8,
        hd_probe_layer: int = 13,
        hd_max: Optional[float] = None,
        orthogonalize: bool = True,
    ) -> None:
        self.model = model

        # ---- resolve the two directions ----------------------------------
        # Priority: explicit RD/HD kwargs > legacy safety_vectors (as RD).
        rd = rejection_direction if rejection_direction is not None else safety_vectors
        if rd is None:
            rd = {}
        rd = {int(k): _to_numpy(v) for k, v in dict(rd).items()}

        if harmfulness_direction is not None:
            hd = {int(k): _to_numpy(v) for k, v in dict(harmfulness_direction).items()}
            if orthogonalize:
                _, hd = extract_adasteer_directions(
                    rejection_pos={k: v for k, v in rd.items()},
                    rejection_neg={k: v * 0.0 for k, v in rd.items()},
                    harm_pos={k: v for k, v in hd.items()},
                    harm_neg={k: v * 0.0 for k, v in hd.items()},
                    orthogonalize=True,
                )
        else:
            # No HD given -> single-law degenerate mode (HD == RD).
            hd = {k: v.copy() for k, v in rd.items()}

        self.rejection_direction: Dict[int, Any] = rd
        self.harmfulness_direction: Dict[int, Any] = hd

        self.config = AdaSteerConfig(
            target_layers=list(target_layers)
            if target_layers is not None
            else list(range(10, 20)),
            rd_probe_layer=rd_probe_layer,
            hd_probe_layer=hd_probe_layer,
            rd_max=base_multiplier,
            hd_max=hd_max if hd_max is not None else base_multiplier,
            min_coeff=0.0,
            adaptive=adaptive,
            orthogonalize=orthogonalize,
        )
        self.safety_threshold = safety_threshold

        self.coeff = AdaSteerCoefficient(
            rd_layer=rd_probe_layer,
            hd_layer=hd_probe_layer,
            rd_max=self.config.rd_max,
            hd_max=self.config.hd_max,
            min_coeff=self.config.min_coeff,
        )

        # Per-forward adaptive state, recomputed for every input batch.
        self._rd_coeff: Any = None  # (N,) tensor of RD coefficients
        self._hd_coeff: Any = None  # (N,) tensor of HD coefficients
        self._captured: Dict[int, Any] = {}  # probe-layer activations
        self._hooks: List[Any] = []

        self.register_hooks()

    # -- layer access -------------------------------------------------------
    def _get_layers(self) -> list:
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return list(m.model.layers)
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return list(m.transformer.h)
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return list(m.gpt_neox.layers)
        return []

    # -- calibration --------------------------------------------------------
    def fit_adaptive(
        self,
        harmful_acts: Dict[int, Any],
        benign_acts: Dict[int, Any],
        rd_anchor: Optional[Any] = None,
        hd_anchor: Optional[Any] = None,
    ) -> "AdaSteerModel":
        """Learn the logistic Rejection/Harmfulness laws from calibration sets.

        ``harmful_acts`` / ``benign_acts`` map ``layer -> (N, D)`` last-token
        activations.  Must include ``rd_probe_layer`` and ``hd_probe_layer``.
        """
        rd_dir = self.rejection_direction.get(self.config.rd_probe_layer)
        hd_dir = self.harmfulness_direction.get(self.config.hd_probe_layer)
        if rd_dir is None or hd_dir is None:
            raise ValueError(
                "AdaSteer.fit_adaptive: directions missing at probe layers "
                f"rd={self.config.rd_probe_layer}, hd={self.config.hd_probe_layer}"
            )
        self.coeff.fit(
            rd_direction=rd_dir,
            hd_direction=hd_dir,
            harmful_acts=harmful_acts,
            benign_acts=benign_acts,
            rd_anchor=rd_anchor,
            hd_anchor=hd_anchor,
        )
        return self

    # -- hooks --------------------------------------------------------------
    def _probe_hook(self, layer_idx: int):
        """Capture the last-token activation at a probe layer (RD or HD)."""

        def hook(module: Any, inp: Any, output: Any):
            hidden = output[0] if isinstance(output, tuple) else output
            # Only the prompt forward (seq_len > 1) carries a meaningful
            # last instruction token; skip per-step decoding forwards.
            if hidden.dim() == 3 and hidden.shape[1] > 1:
                self._captured[layer_idx] = hidden[:, -1, :].detach()
            return output

        return hook

    def _compute_coefficients(self) -> None:
        """Recompute per-input RD/HD coefficients from captured activations.

        This is the adaptive step: it runs *inside* the forward path (driven
        by ``_probe_hook`` captures) so the steering applied below is genuinely
        input-dependent, not a fixed multiplier.
        """

        rd_layer = self.config.rd_probe_layer
        hd_layer = self.config.hd_probe_layer
        rd_dir = self.rejection_direction.get(rd_layer)
        hd_dir = self.harmfulness_direction.get(hd_layer)

        if rd_layer in self._captured and rd_dir is not None:
            proj = self.coeff._project(self._captured[rd_layer], rd_dir)
            self._rd_coeff = self.coeff.rd_coefficient(proj)
        if hd_layer in self._captured and hd_dir is not None and self.config.adaptive:
            proj = self.coeff._project(self._captured[hd_layer], hd_dir)
            self._hd_coeff = self.coeff.hd_coefficient(proj)

    def _steer_hook(self, layer_idx: int):
        """Add ``c_RD * RD + c_HD * HD`` to the residual stream at a layer."""

        def hook(module: Any, inp: Any, output: Any):
            try:
                import torch
            except ImportError:  # pragma: no cover
                return output

            hidden = output[0] if isinstance(output, tuple) else output

            # Make sure adaptive coefficients reflect *this* input.  Probe
            # layers precede every target layer, so captures are ready.
            if self._rd_coeff is None and self._hd_coeff is None:
                self._compute_coefficients()

            bsz = hidden.shape[0]
            device, dtype = hidden.device, hidden.dtype
            delta = torch.zeros_like(hidden)

            rd = self.rejection_direction.get(layer_idx)
            if rd is not None:
                rd_t = torch.as_tensor(rd, device=device, dtype=dtype)
                if self.config.adaptive and self._rd_coeff is not None:
                    c = torch.as_tensor(
                        self._rd_coeff, device=device, dtype=dtype
                    ).reshape(-1)[:bsz]
                else:
                    c = torch.full((bsz,), float(self.config.rd_max),
                                   device=device, dtype=dtype)
                delta = delta + c.view(bsz, 1, 1) * rd_t.view(1, 1, -1)

            if self.config.adaptive:
                hd = self.harmfulness_direction.get(layer_idx)
                if hd is not None and self._hd_coeff is not None:
                    hd_t = torch.as_tensor(hd, device=device, dtype=dtype)
                    c = torch.as_tensor(
                        self._hd_coeff, device=device, dtype=dtype
                    ).reshape(-1)[:bsz]
                    delta = delta + c.view(bsz, 1, 1) * hd_t.view(1, 1, -1)

            hidden = hidden + delta
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden

        return hook

    def register_hooks(self) -> None:
        """Install probe hooks (RD/HD layers) and steering hooks (targets)."""
        self.remove_hooks()
        layers = self._get_layers()
        n = len(layers)

        for idx in {self.config.rd_probe_layer, self.config.hd_probe_layer}:
            if 0 <= idx < n:
                self._hooks.append(
                    layers[idx].register_forward_hook(self._probe_hook(idx))
                )
        for idx in self.config.target_layers:
            if 0 <= idx < n:
                self._hooks.append(
                    layers[idx].register_forward_hook(self._steer_hook(idx))
                )
        logger.info(
            "AdaSteer: %d hooks (probe + steer) over %d target layers.",
            len(self._hooks),
            len(self.config.target_layers),
        )

    def remove_hooks(self) -> None:
        for h in getattr(self, "_hooks", []):
            h.remove()
        self._hooks = []

    def _reset_adaptive_state(self) -> None:
        """Clear per-input adaptive state so the next input is re-evaluated."""
        self._rd_coeff = None
        self._hd_coeff = None
        self._captured = {}

    # -- legacy / manual override ------------------------------------------
    def set_adaptive_multiplier(self, safety_score: float) -> None:
        """Back-compat: pin the RD coefficient from an external safety score.

        Retained so old callers do not break.  The normal AdaSteer path is
        fully automatic -- coefficients are recomputed per input inside the
        forward pass via the logistic laws; calling this is not required.
        """
        import numpy as np

        score = float(safety_score)
        if score < self.safety_threshold:
            scale = 1.0 + (self.safety_threshold - score)
        else:
            scale = max(0.1, 1.0 - (score - self.safety_threshold))
        self._rd_coeff = np.asarray([self.config.rd_max * scale], dtype=np.float64)

    # -- generation ---------------------------------------------------------
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        self._reset_adaptive_state()
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._reset_adaptive_state()
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        return getattr(self.__dict__["model"], name)

    def __enter__(self) -> "AdaSteerModel":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "AdaSteerModel":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(path)
        return cls(model, **kwargs)
