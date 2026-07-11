"""AlphaSteer inference-time steering model wrapper.

Faithful local implementation of:

    "AlphaSteer: Learning Refusal Steering with Principled Null-Space
    Constraint", arXiv:2506.07022 (ICLR 2026).
    Original repo: https://github.com/AlphaLab-USTC/AlphaSteer

AlphaSteer learns, *per steered layer*, a steering transform that

  (1) **preserves utility** by being constrained to the null space of the
      benign activation matrix -- so benign activations are (nearly) unsteered;
  (2) **enhances safety** by regressing malicious activations onto an explicit
      *refusal direction* via ridge-regularized least squares.

This module previously delegated to ``safetune.core.runtime.inference.alphasteer``,
whose ``fit`` used a crude ``Y = -X_h`` regression target and a single fixed
layer.  The audit (``audit_faithfulness/steer.md``, rated simplified-correct)
flagged that shortcut.  The logic below is now implemented locally and follows
the authors' reference code (``src/utils/steering_utils.py`` and
``src/calc_steering_matrix.py``) exactly.

Authors' algorithm, traced to the reference repo
------------------------------------------------
For one layer, with benign activations ``H_b`` (shape ``[N_b, d]``), malicious
activations ``H_m`` (shape ``[N_m, d]``) and a refusal direction ``r`` (``[d]``):

* Null-space projector  -- ``null_space_l`` / ``null_space_projection_l``::

      _, S, Vh = svd(H_b.T @ H_b)            # non-central covariance, [d, d]
      num     = int(d * abs_nullspace_ratio) # smallest-singular-value count
      Q       = Vh[-num:, :].T               # null-space orthonormal basis
      P       = Q @ Q.T                      # projector onto null space, [d, d]

* Ridge-regularized regression -- ``cal_tilde_delta_with_regularization_l``::

      X            = H_m @ P                 # [N_m, d]
      A            = X.T @ X + lambda_reg * (P.T @ P)   # [d, d]
      b            = X.T @ r.repeat(N_m, 1)  # [d, N_m] @ [N_m, d] -> [d, d]
      tilde_delta  = pinv(A) @ b             # [d, d]

* Steering matrix -- ``cal_steering_matrix_l``::

      steering_matrix = P @ tilde_delta      # [d, d]  (per steered layer)

* Inference -- ``AlphaLlamaDecoderLayer.forward``: at the *last valid token*
  position only, on the prefill pass, add
  ``last_hidden @ steering_matrix * strength`` to the residual stream of every
  token position of that layer.

The public ``AlphaSteerModel.__init__`` signature is unchanged; the extra
needs of the faithful implementation are exposed as optional keyword arguments
with backward-compatible defaults.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

try:  # torch is required for the real implementation; degrade gracefully.
    import torch

    _TORCH_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_ERROR = _e


# --------------------------------------------------------------------------
# Core math -- direct ports of the authors' src/utils/steering_utils.py
# --------------------------------------------------------------------------
def _null_space_basis(
    activations: "torch.Tensor",
    abs_nullspace_ratio: float = 0.6,
    min_null_space_ratio: float = 0.1,
    abs_null_rank: Optional[int] = None,
) -> "torch.Tensor":
    """Orthonormal basis ``Q`` (``[d, k]``) for the null space of ``activations``.

    Port of ``null_space_l``.  ``activations`` has shape ``[N, d]``; the SVD is
    taken of the ``[d, d]`` non-central covariance ``Aᵀ A``.  The null space is
    spanned by the singular vectors with the *smallest* singular values, i.e.
    the trailing columns of ``Vhᵀ``.

    Selection of the null-space dimension ``k`` (priority order):
      * ``abs_null_rank`` -- an explicit integer rank (``null_rank`` kwarg);
      * ``abs_nullspace_ratio > 0`` -- ``k = int(d * ratio)`` (authors' path);
      * otherwise a numerical-tolerance count, floored at ``min_null_space_ratio``.
    """
    A = activations
    M, N = A.shape[0], A.shape[1]
    # svd of the non-central covariance; S is descending.
    _, S, Vh = torch.linalg.svd(A.T @ A)

    if abs_null_rank is not None:
        num = int(abs_null_rank)
    elif abs_nullspace_ratio and abs_nullspace_ratio > 0:
        num = int(N * abs_nullspace_ratio)
    else:
        S_ = torch.sqrt(S.clamp_min(0))
        rcond = torch.finfo(S.dtype).eps * max(M, N)
        tol = torch.amax(S_) * rcond
        num = int(torch.sum(S_ < tol).item())
        if num / N < min_null_space_ratio:
            num = int(N * min_null_space_ratio)

    num = max(0, min(num, N))
    if num == 0:
        # Degenerate: empty basis -> zero projector downstream.
        return Vh.new_zeros((N, 0))
    # Trailing `num` rows of Vh -> smallest singular values -> null space.
    Q = Vh[-num:, :].T.conj()
    return Q


def _null_space_projection(
    activations: "torch.Tensor",
    abs_nullspace_ratio: float = 0.6,
    min_null_space_ratio: float = 0.1,
    abs_null_rank: Optional[int] = None,
) -> "torch.Tensor":
    """Projector ``P = Q Qᵀ`` (``[d, d]``) onto the benign null space.

    Port of ``null_space_projection_l``.
    """
    Q = _null_space_basis(
        activations,
        abs_nullspace_ratio=abs_nullspace_ratio,
        min_null_space_ratio=min_null_space_ratio,
        abs_null_rank=abs_null_rank,
    )
    d = activations.shape[1]
    if Q.shape[1] == 0:
        return activations.new_zeros((d, d))
    return Q @ Q.T


def _tilde_delta_regularized(
    harmful: "torch.Tensor",
    P: "torch.Tensor",
    refusal_vector: "torch.Tensor",
    lambda_reg: float,
) -> "torch.Tensor":
    """Ridge-regularized regression solution ``tilde_delta`` (``[d, d]``).

    Port of ``cal_tilde_delta_with_regularization_l``.  Solves, in the benign
    null space, for the transform that maps every malicious activation onto the
    refusal direction ``r``::

        X           = H_m @ P                          # [N_m, d]
        A           = Xᵀ X + lambda_reg * (Pᵀ P)       # [d, d]
        b           = Xᵀ r.repeat(N_m, 1)              # [d, d]
        tilde_delta = pinv(A) @ b                      # [d, d]

    ``r.repeat(N_m, 1)`` stacks the same refusal direction ``N_m`` times, so the
    regression target is identical for every malicious activation -- as in the
    authors' code.
    """
    X = harmful @ P                                  # [N_m, d]
    A = X.T @ X + lambda_reg * (P.T @ P)             # [d, d]
    target = refusal_vector.unsqueeze(0).expand(X.shape[0], -1)  # [N_m, d]
    b = X.T @ target                                 # [d, d]
    tilde_delta = torch.linalg.pinv(A) @ b           # [d, d]
    return tilde_delta


def _refusal_direction(
    harmful: "torch.Tensor", benign: "torch.Tensor"
) -> "torch.Tensor":
    """Default refusal direction when the caller does not supply ``targets``.

    AlphaSteer regresses malicious activations onto an explicit *refusal
    direction* (the authors load a precomputed ``RV_refusal`` vector per
    layer).  When none is given we fall back to the standard refusal-direction
    estimate -- the diff-of-means ``mean(harmful) - mean(benign)`` -- which is
    the contrast-pair direction AlphaSteer's refusal-vector extraction is built
    on.  This replaces the previous crude ``Y = -X_h`` target.
    """
    return harmful.mean(dim=0) - benign.mean(dim=0)


def _fit_layer(
    harmful: "torch.Tensor",
    benign: "torch.Tensor",
    refusal_vector: "torch.Tensor",
    lambda_reg: float,
    abs_nullspace_ratio: float,
    abs_null_rank: Optional[int],
) -> "torch.Tensor":
    """Fit one layer's steering matrix ``steering = P @ tilde_delta`` (``[d, d]``)."""
    P = _null_space_projection(
        benign,
        abs_nullspace_ratio=abs_nullspace_ratio,
        abs_null_rank=abs_null_rank,
    )
    tilde_delta = _tilde_delta_regularized(harmful, P, refusal_vector, lambda_reg)
    # cal_steering_matrix_l: steering_matrix = P @ tilde_delta
    return P @ tilde_delta


class AlphaSteerModel:
    """Inference-time AlphaSteer wrapper.

    Public ``__init__`` parameters (signature preserved):
        model               -- a HF causal-LM (or compatible) module.
        harmful_activations -- malicious activations, ``[N, d]`` or ``[N, L, d]``.
        benign_activations  -- benign activations, same layout as ``harmful``.
        targets             -- explicit refusal direction(s): ``[d]`` or
                               ``[L, d]``.  ``None`` -> diff-of-means default.
        layer_id            -- single steered layer when activations are 2-D
                               (and the default when ``layers`` is unset).
        lambda_ridge        -- ridge regularization strength ``lambda_reg``.
        null_rank           -- explicit absolute null-space rank ``k``; overrides
                               ``nullspace_ratio`` when given.

    Faithfulness extras (optional, defaulted -- signature still compatible):
        layers              -- explicit list of layer indices to steer; when
                               activations are 3-D this defaults to *every*
                               layer present, matching the authors' multi-layer
                               steering.
        nullspace_ratio     -- authors' ``abs_nullspace_ratio`` (default 0.6).
        strength            -- per-layer steering coefficient ``lambda`` applied
                               at inference (scalar or per-layer sequence).
        last_token_only     -- apply steering using the last-valid-token hidden
                               state (authors' behavior) vs. per-position.
        prefill_only        -- apply steering only on the prefill pass (the
                               authors steer the prompt, not each decoded token).
    """

    def __init__(
        self,
        model: Any,
        harmful_activations: Any,
        benign_activations: Any,
        targets: Any = None,
        layer_id: int = 15,
        lambda_ridge: float = 10.0,  # matches the official AlphaSteer release (was 1e-4)
        null_rank: Optional[int] = None,
        *,
        layers: Optional[Sequence[int]] = None,
        nullspace_ratio: float = 0.6,
        strength: Union[float, Sequence[float]] = 1.0,
        last_token_only: bool = True,
        prefill_only: bool = True,
    ) -> None:
        if _TORCH_ERROR is not None:  # pragma: no cover
            raise ImportError("AlphaSteerModel requires torch") from _TORCH_ERROR

        self.model = model
        self.layer_id = int(layer_id)
        self.lambda_ridge = float(lambda_ridge)
        self.null_rank = null_rank
        self.nullspace_ratio = float(nullspace_ratio)
        self.last_token_only = bool(last_token_only)
        self.prefill_only = bool(prefill_only)
        self._hooks: List[Any] = []

        H_h = self._as_tensor(harmful_activations)
        H_b = self._as_tensor(benign_activations)

        # Normalize to a per-layer dict {layer_idx: steering_vector [d]}.
        self.steering_matrices: dict[int, "torch.Tensor"] = {}

        if H_h.dim() == 2:
            # Single-layer activations -> steer only `layer_id`.
            steer_layers = list(layers) if layers is not None else [self.layer_id]
            if len(steer_layers) != 1:
                raise ValueError(
                    "2-D activations describe a single layer; pass 3-D "
                    "activations [N, L, d] to steer multiple layers."
                )
            r = self._refusal_for(targets, H_h, H_b, 0, single=True)
            self.steering_matrices[steer_layers[0]] = _fit_layer(
                H_h, H_b, r, self.lambda_ridge, self.nullspace_ratio, self.null_rank
            )
        elif H_h.dim() == 3:
            # Multi-layer activations [N, L, d].
            n_layers = H_h.shape[1]
            steer_layers = (
                list(layers) if layers is not None else list(range(n_layers))
            )
            for li in steer_layers:
                if not 0 <= li < n_layers:
                    raise ValueError(
                        f"layer index {li} out of range for {n_layers} layers"
                    )
                r = self._refusal_for(targets, H_h[:, li, :], H_b[:, li, :], li)
                self.steering_matrices[li] = _fit_layer(
                    H_h[:, li, :],
                    H_b[:, li, :],
                    r,
                    self.lambda_ridge,
                    self.nullspace_ratio,
                    self.null_rank,
                )
        else:
            raise ValueError(
                "activations must be 2-D [N, d] or 3-D [N, L, d]; "
                f"got shape {tuple(H_h.shape)}"
            )

        # Per-layer steering coefficient(s).
        if isinstance(strength, (int, float)):
            self.strengths = {li: float(strength) for li in self.steering_matrices}
        else:
            s = list(strength)
            self.strengths = {
                li: float(s[i]) for i, li in enumerate(sorted(self.steering_matrices))
            }

        self.register_hooks()

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _as_tensor(x: Any) -> "torch.Tensor":
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        return x.detach().float()

    def _refusal_for(
        self,
        targets: Any,
        harmful: "torch.Tensor",
        benign: "torch.Tensor",
        layer_idx: int,
        single: bool = False,
    ) -> "torch.Tensor":
        """Resolve the refusal direction for one layer.

        ``targets`` may be ``None`` (diff-of-means default), a single ``[d]``
        vector, or a per-layer ``[L, d]`` stack.
        """
        if targets is None:
            return _refusal_direction(harmful, benign)
        t = self._as_tensor(targets)
        if t.dim() == 1:
            return t
        if t.dim() == 2:
            if single:
                # 2-D targets with single-layer activations: treat as either a
                # [L, d] stack indexed by layer_id, or a [1, d] vector.
                return t[self.layer_id] if t.shape[0] > self.layer_id else t[0]
            return t[layer_idx]
        raise ValueError(f"targets must be 1-D or 2-D; got {tuple(t.shape)}")

    def _resolve_layers(self) -> Any:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        if hasattr(self.model, "layers"):
            return self.model.layers
        raise AttributeError("Could not locate transformer layers on model")

    @staticmethod
    def _last_valid_index(
        attention_mask: Any, batch: int, seq_len: int, device: Any
    ) -> "torch.Tensor":
        """Index of the last non-pad token per sequence (robust to L/R padding)."""
        if attention_mask is None:
            return torch.full((batch,), seq_len - 1, dtype=torch.long, device=device)
        mask = attention_mask.to(device)
        if mask.dim() > 2:  # collapse extended (4-D) masks to [B, T]
            mask = mask.reshape(batch, -1)[:, :seq_len]
        # last position where mask != 0
        idx = (mask != 0).float()
        # argmax of reversed -> last True; fall back to seq_len-1 if all zero.
        rev = torch.flip(idx, dims=[1])
        last = seq_len - 1 - torch.argmax(rev, dim=1)
        last = torch.where(idx.sum(dim=1) > 0, last, torch.full_like(last, seq_len - 1))
        return last.to(torch.long)

    def _make_hook(self, layer_idx: int):
        steering = self.steering_matrices[layer_idx]
        strength = self.strengths.get(layer_idx, 1.0)

        def hook(module: Any, inputs: Any, output: Any) -> Any:
            if strength == 0.0:
                return output
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            if not torch.is_tensor(hidden) or hidden.dim() != 3:
                return output

            B, T, D = hidden.shape
            # Authors steer the prompt (prefill), not individual decoded tokens.
            if self.prefill_only and T <= 1:
                return output

            sm = steering.to(device=hidden.device, dtype=hidden.dtype)

            if self.last_token_only:
                # Build the steering vector from the last valid token's hidden
                # state and broadcast it across every position (authors'
                # AlphaLlamaDecoderLayer.forward).
                attn = None
                if isinstance(inputs, (tuple, list)) and len(inputs) > 1:
                    attn = inputs[1]
                last_idx = self._last_valid_index(attn, B, T, hidden.device)
                batch_idx = torch.arange(B, device=hidden.device)
                last_hidden = hidden[batch_idx, last_idx, :]      # [B, D]
                steer_vec = (last_hidden @ sm) * strength         # [B, D]
                hidden = hidden + steer_vec.unsqueeze(1)
            else:
                # Per-position variant: steer every token by its own state.
                hidden = hidden + (hidden @ sm) * strength

            return (hidden,) + tuple(output[1:]) if is_tuple else hidden

        return hook

    # ---- public API ------------------------------------------------------
    def register_hooks(self) -> None:
        self.remove_hooks()
        layers = self._resolve_layers()
        for li in self.steering_matrices:
            if not 0 <= li < len(layers):
                continue
            self._hooks.append(
                layers[li].register_forward_hook(self._make_hook(li))
            )

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        return getattr(self.__dict__["model"], name)

    def __enter__(self) -> "AlphaSteerModel":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "AlphaSteerModel":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(path)
        return cls(model, **kwargs)
