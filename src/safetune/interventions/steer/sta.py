"""STA (Steering Target Atoms) inference-time model wrapper.

Paper
-----
"Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering
Target Atoms", Wang, Xu, Mao, Deng, Tu, Chen, Zhang. ACL 2025,
arXiv:2505.20322. Original repo: https://github.com/zjunlp/steer-target-atoms

Genuine STA algorithm (the unit of intervention is an SAE latent feature)
-------------------------------------------------------------------------
STA does *not* steer attention heads or arbitrary residual-stream slices.
Its "target atoms" are individual *Sparse Autoencoder (SAE) latent features*
-- disentangled, monosemantic directions in the SAE's high-dimensional
(M >> D) latent space. The method:

1.  An SAE is trained on a layer's residual-stream activations ``h in R^D``.
    It encodes  ``a = JumpReLU(h W_enc + b_enc) in R^M``  (M >> D, sparse)
    and decodes ``h_hat = a W_dec + b_dec``.

2.  "Target atoms" are *specific latent dimensions* ``j in {0..M-1}`` of the
    SAE that are causally responsible for the target behaviour.

3.  Atoms are *identified* by contrasting positive vs negative responses in
    SAE latent space ("act and fre" selection):
      * activation score   ``Delta_a[j] = mean(a_pos[:, j]) - mean(a_neg[:, j])``
      * frequency score    ``Delta_f[j] = freq(a_pos[:, j] > 0) - freq(a_neg[:, j] > 0)``
    After sign-consistency masking and top-percentile trimming, the kept
    atoms form a sparse feature vector ``a_target in R^M``.

4.  The steering vector is the SAE *decoder* projection of the trimmed atom
    vector back into the residual stream:  ``v_STA = a_target @ W_dec``.

5.  At inference STA adds ``h_hat = h + lambda * v_STA`` to the residual
    stream at the SAE's layer.

This module implements that interface faithfully. A genuine SAE (e.g. a
``sae_lens.SAE`` or a Gemma-Scope SAE) must be supplied by the caller; if no
SAE-backed steering vector is available the wrapper degrades gracefully and
documents the requirement rather than fabricating arbitrary slice steering.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

logger = logging.getLogger(__name__)

try:
    from safetune.core.runtime.inference.sta import (
        STAConfig as _CoreSTAConfig,
        STAWrapper,
    )
    _IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    STAWrapper = None  # type: ignore[assignment]
    _CoreSTAConfig = None  # type: ignore[assignment]
    _IMPORT_ERROR = _e


@runtime_checkable
class SAEProtocol(Protocol):
    """Minimal interface STA expects from a Sparse Autoencoder.

    Any object exposing an ``encode``/``decode`` pair and a decoder weight
    matrix ``W_dec`` of shape ``(M, D)`` satisfies this protocol. The widely
    used ``sae_lens.SAE`` class (Gemma-Scope, Llama-Scope, etc.) conforms
    directly. ``hook_layer`` (or ``cfg.hook_layer``) tells STA which decoder
    layer the SAE was trained on.
    """

    def encode(self, acts: Any) -> Any:  # h (.., D) -> a (.., M), JumpReLU sparse
        ...

    def decode(self, latents: Any) -> Any:  # a (.., M) -> h_hat (.., D)
        ...


def _sae_layer(sae: Any, default: int) -> int:
    """Best-effort extraction of the layer index an SAE was trained on."""
    for attr in ("hook_layer", "layer"):
        if hasattr(sae, attr):
            try:
                return int(getattr(sae, attr))
            except Exception:  # pragma: no cover
                pass
    cfg = getattr(sae, "cfg", None)
    if cfg is not None:
        for attr in ("hook_layer", "layer"):
            if hasattr(cfg, attr):
                try:
                    return int(getattr(cfg, attr))
                except Exception:  # pragma: no cover
                    pass
        hook_name = getattr(cfg, "hook_name", "") or ""
        # e.g. "blocks.20.hook_resid_post"
        for part in str(hook_name).split("."):
            if part.isdigit():
                return int(part)
    return default


def select_target_atoms(
    pos_latents: Any,
    neg_latents: Any,
    keep_fraction: float = 0.05,
) -> Tuple[Any, Any]:
    """Identify STA target atoms from contrastive SAE latent activations.

    This implements the paper's "act and fre" selection (see
    ``sae_feature_selection.py`` / ``act_and_fre`` in the original repo).

    Args
    ----
    pos_latents : tensor ``(N_pos, M)`` -- SAE latent activations ``a`` for
        positive (target-behaviour) responses, mean-pooled per example.
    neg_latents : tensor ``(N_neg, M)`` -- SAE latent activations for the
        contrasting negative responses.
    keep_fraction : top fraction of latent dimensions to keep (paper's
        ``trim``; the repo default selects the top ~5%).

    Returns
    -------
    (atom_scores, atom_indices)
        ``atom_scores`` is a length-M sparse vector with the contrastive
        activation score on the kept atoms and 0 elsewhere; ``atom_indices``
        are the kept latent-feature (atom) indices.
    """
    import torch

    pos = pos_latents.float()
    neg = neg_latents.float()

    # Contrastive activation amplitude:  Delta_a = mean(a_pos) - mean(a_neg)
    act_score = pos.mean(0) - neg.mean(0)
    # Contrastive activation frequency:  Delta_f = freq_pos - freq_neg
    freq_score = (pos > 0).float().mean(0) - (neg > 0).float().mean(0)

    def _signed_min_max(t: "torch.Tensor") -> "torch.Tensor":
        a = t.abs()
        lo, hi = a.min(), a.max()
        denom = (hi - lo).clamp_min(1e-12)
        return t.sign() * ((a - lo) / denom)

    norm_act = _signed_min_max(act_score)
    norm_freq = _signed_min_max(freq_score)

    # Sign-consistency: keep atoms whose activation and frequency agree.
    sign_mask = ((norm_act > 0) & (norm_freq > 0)) | ((norm_act < 0) & (norm_freq < 0))
    freq_ranking = torch.zeros_like(norm_freq)
    freq_ranking[sign_mask] = norm_freq[sign_mask]

    m = act_score.numel()
    k = max(1, min(m, int(keep_fraction * m)))

    # Frequency top-k and activation-magnitude top-k; STA keeps the intersection.
    sorted_freq = torch.sort(freq_ranking.abs(), descending=True).values
    freq_thr = sorted_freq[min(k, m - 1)]
    freq_mask = freq_ranking.abs() >= freq_thr

    sorted_act = torch.sort(act_score.abs(), descending=True).values
    act_thr = sorted_act[min(k, m - 1)]
    act_mask = act_score.abs() >= act_thr

    combined_mask = freq_mask & act_mask
    if combined_mask.sum() == 0:  # fallback: do not return an empty atom set
        combined_mask = act_mask

    atom_scores = torch.zeros_like(act_score)
    atom_scores[combined_mask] = act_score[combined_mask]
    atom_indices = combined_mask.nonzero(as_tuple=True)[0]
    return atom_scores, atom_indices


def build_sta_steering_vector(
    sae: SAEProtocol,
    atom_scores: Any,
) -> Any:
    """Decode a sparse target-atom vector into a residual-stream steering vector.

    Implements ``v_STA = a_target @ W_dec`` (the paper's Eq. for the steering
    vector; ``steering_vector = feature_score @ sae.W_dec`` in the repo).
    """

    scores = atom_scores
    if hasattr(sae, "W_dec") and getattr(sae, "W_dec") is not None:
        w_dec = sae.W_dec
        scores = scores.to(dtype=w_dec.dtype, device=w_dec.device)
        return scores @ w_dec
    # Fall back to the SAE's own decode (handles bias / non-linear decoders).
    return sae.decode(scores)


class STAModel:
    """Apply genuine STA SAE-latent-feature steering to a base model.

    The public signature is preserved for backward compatibility, but the
    arguments now carry their *correct* STA semantics:

    * ``atom_vectors`` -- mapping ``layer_idx -> steering vector`` already
      decoded from SAE atoms (i.e. ``v_STA`` per layer). The legacy
      ``(layer, head)`` key form is still accepted; the head component is
      ignored because STA atoms are SAE features, not attention heads.
    * ``target_atoms`` -- when an ``sae`` is supplied, this is the list of
      *SAE latent feature indices* (the actual "target atoms"); the steering
      vector is then built from the SAE decoder. The legacy ``(layer, head)``
      tuple form is also accepted (head ignored).

    New optional, defaulted keyword arguments (no signature break):

    * ``sae`` -- a Sparse Autoencoder satisfying :class:`SAEProtocol`. When
      given together with ``atom_scores`` (or ``target_atoms`` as feature
      indices), STA decodes the genuine SAE-atom steering vector.
    * ``atom_scores`` -- length-M sparse latent vector (output of
      :func:`select_target_atoms`); decoded via the SAE into ``v_STA``.
    * ``sae_layer`` -- override for the residual-stream layer to steer
      (defaults to the SAE's own ``hook_layer``).
    """

    def __init__(
        self,
        model: Any,
        atom_vectors: Optional[Dict[Any, Any]] = None,
        target_atoms: Optional[Union[List[int], List[Tuple[int, int]]]] = None,
        multiplier: float = 3.0,
        *,
        sae: Optional[SAEProtocol] = None,
        atom_scores: Optional[Any] = None,
        sae_layer: Optional[int] = None,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.runtime.inference.sta is unavailable"
            ) from _IMPORT_ERROR

        self.model = model
        self.sae = sae
        self.multiplier = float(multiplier)

        # ------------------------------------------------------------------
        # Resolve per-layer STA steering vectors v_STA (residual-stream R^D).
        # ------------------------------------------------------------------
        layer_vectors: Dict[int, Any] = {}

        if sae is not None and atom_scores is not None:
            # Genuine STA path: decode the sparse SAE-atom vector via W_dec.
            layer = sae_layer if sae_layer is not None else _sae_layer(sae, default=20)
            layer_vectors[int(layer)] = build_sta_steering_vector(sae, atom_scores)
            logger.info(
                "STA: built SAE-atom steering vector for layer %d "
                "(%d non-zero atoms).",
                layer,
                int((atom_scores != 0).sum()) if hasattr(atom_scores, "sum") else -1,
            )
        elif sae is not None and target_atoms is not None and _is_index_list(target_atoms):
            # target_atoms given as SAE latent feature indices -> unit-weight atoms.
            import torch

            w_dec = getattr(sae, "W_dec", None)
            m = w_dec.shape[0] if w_dec is not None else None
            if m is None:
                raise ValueError(
                    "STA: cannot infer SAE latent width; pass `atom_scores` "
                    "(a length-M sparse vector) instead of bare indices."
                )
            scores = torch.zeros(m)
            for idx in target_atoms:  # type: ignore[union-attr]
                scores[int(idx)] = 1.0
            layer = sae_layer if sae_layer is not None else _sae_layer(sae, default=20)
            layer_vectors[int(layer)] = build_sta_steering_vector(sae, scores)
            logger.info(
                "STA: built SAE-atom steering vector for layer %d from %d "
                "feature indices.",
                layer,
                len(target_atoms),  # type: ignore[arg-type]
            )
        elif atom_vectors:
            # Caller supplied pre-decoded v_STA vectors directly.
            for key, vec in atom_vectors.items():
                layer_idx = _layer_of(key)
                if layer_idx in layer_vectors:
                    # Multiple atoms on the same layer: sum their v_STA
                    # contributions (decoding is linear in the atom vector).
                    layer_vectors[layer_idx] = layer_vectors[layer_idx] + vec
                else:
                    layer_vectors[layer_idx] = vec
            logger.info(
                "STA: using %d caller-supplied steering vectors.",
                len(layer_vectors),
            )
        else:
            logger.warning(
                "STA: no SAE and no steering vectors supplied -- STAModel is "
                "inert. STA's target atoms are SAE latent features; supply "
                "either `sae` + `atom_scores` (see `select_target_atoms`) or "
                "pre-decoded `atom_vectors` keyed by layer index. STA does NOT "
                "steer attention heads or arbitrary residual-stream slices."
            )

        self.layer_vectors = layer_vectors

        # Keep a (configured but un-hooked) core STAWrapper for API parity /
        # introspection. STA steering itself is applied by this class's own
        # full-residual-stream additive hooks below: STA's atoms are SAE
        # latent features whose decoded v_STA is added to the *whole* D-dim
        # residual stream -- not sliced into a fake per-head sub-range as the
        # legacy core hook would do. We therefore do NOT call
        # ``self._impl.register_hooks()``.
        cfg = _CoreSTAConfig(
            target_atoms=[(int(l), 0) for l in layer_vectors],
            multiplier=self.multiplier,
        )
        self._impl = STAWrapper(
            model=model,
            atom_vectors={(int(l), 0): v for l, v in layer_vectors.items()},
            config=cfg,
        )

        self._hooks: List[Any] = []
        self._install_hooks()

    # ----------------------------------------------------------------------
    # STA residual-stream steering hooks (h_hat = h + lambda * v_STA).
    # ----------------------------------------------------------------------
    def _get_layers(self) -> list:
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return list(m.model.layers)
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return list(m.transformer.h)
        return []

    def _make_hook(self, vector: Any):
        mult = self.multiplier

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            sv = vector.to(device=hidden.device, dtype=hidden.dtype)
            if sv.shape[-1] != hidden.shape[-1]:
                logger.warning(
                    "STA: steering vector dim %d != residual-stream dim %d; "
                    "skipping (SAE/model mismatch).",
                    sv.shape[-1],
                    hidden.shape[-1],
                )
                return output
            # Add the SAE-atom-decoded steering vector to the full residual
            # stream at every token position.
            hidden = hidden + mult * sv
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden

        return hook

    def _install_hooks(self) -> None:
        self.remove_hooks()
        layers = self._get_layers()
        for layer_idx, vec in self.layer_vectors.items():
            if 0 <= layer_idx < len(layers):
                handle = layers[layer_idx].register_forward_hook(
                    self._make_hook(vec)
                )
                self._hooks.append(handle)
        logger.info("STA: registered SAE-atom steering hooks on %d layers.",
                    len(self._hooks))

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        return getattr(self.__dict__["model"], name)

    def remove_hooks(self) -> None:
        for handle in getattr(self, "_hooks", []):
            try:
                handle.remove()
            except Exception:  # pragma: no cover
                pass
        self._hooks = []
        if hasattr(self._impl, "remove_hooks"):
            self._impl.remove_hooks()

    def __enter__(self) -> "STAModel":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "STAModel":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(path)
        return cls(model, **kwargs)


def _layer_of(key: Any) -> int:
    """Extract the layer index from an atom key (int or (layer, head) tuple)."""
    if isinstance(key, (tuple, list)):
        return int(key[0])
    return int(key)


def _is_index_list(atoms: Sequence[Any]) -> bool:
    """True if `atoms` is a flat list of integer SAE feature indices."""
    if not atoms:
        return False
    return all(isinstance(a, int) for a in atoms)
