"""RepBend — Representation Bending for LLM Safety.

Paper  : "Representation Bending for Large Language Model Safety",
         Yousefpour, Kim, Kwon, Lee, Jeung, Han, Wan, Ngan, et al.,
         ACL 2025, arXiv:2504.01550.
Repo   : https://github.com/AIM-Intelligence/RepBend
         (loss in ``methods/rep_bending/trainer.py`` — function ``_calc_loss``)

WHAT REPBEND ACTUALLY IS
------------------------
RepBend is a **training-time fine-tuning** method, not an inference-time
steering method.  It LoRA-fine-tunes a model with a five-term loss measured on
intermediate hidden states (the residual stream, "h_i4" — the output of each
transformer block), comparing the *fine-tuned* model ``M'`` against the
*frozen original* model ``M``.  The paper's own framing:

    "RepBend brings the idea of activation steering — simple vector
     arithmetic for steering model's behavior during inference — to
     loss-based fine-tuning."

The trained weights intrinsically bend harmful representations; there is **no
runtime hook, no activation edit, and no logit edit** in the paper.

The verbatim loss aggregation from ``_calc_loss`` (RepBend repo) is::

    loss = alpha * safe_loss          # keep safe reps close to original M
         - beta  * unsafe_loss        # push unsafe reps FAR from original M
         + gamma * cosine_loss        # cluster unsafe reps together
         + eps   * kl_loss            # retain general capability (KL to M)
         + eta   * safe_unsafe_loss   # bend unsafe-prompt responses toward safe

where, with ``M'`` = fine-tuned (LoRA) model and ``M`` = frozen original:

  * ``v_s = M'(safe) - M(safe)``  and  ``safe_loss   = mean ||v_s||_2``
  * ``v_u = M'(unsafe) - M(unsafe)`` and ``unsafe_loss = mean ||v_u||_2``
        (subtracted with a negative sign, i.e. *maximised* — push M' far)
  * ``cosine_loss`` = mean of ``(1 - cos_sim)`` over all *pairs* of unsafe
        representations in the batch (encourage harmful reps to collapse
        together, so the model consistently refuses them)
  * ``kl_loss`` = temperature-scaled ``KL( M'(retain) || M(retain) )`` on the
        retain (benign instruction-following) set — capability preservation
  * ``safe_unsafe_loss`` = mean ``||M'(unsafe-request response)
        - M(unsafe-request *safe* response)||_2`` — bends the response region
        of unsafe prompts toward a safe-response trajectory.

WHY THIS WRAPPER EXPOSES THE TRAINING LOSS (NOT A FAKE INFERENCE HOOK)
----------------------------------------------------------------------
The faithfulness audit (``audit_faithfulness/steer.md``) rated the previous
``RepBendModel`` 🟡 and noted it was effectively inert: it registered forward
hooks that only *captured* activations and applied nothing to generations —
"presenting it as an inference-time steerer is a category error."

A faithful *inference-time* analogue of RepBend genuinely does not exist.  The
entire method IS the LoRA weight update produced by the loss above; there is
nothing to "hook" at generation time without inventing an unjustified
heuristic.  Per the project's honesty policy (and mirroring the Circuit
Breakers fix), the correct move is to **expose the real training-time
objective** rather than dress up an inert capture hook as a steering pass.

``RepBendModel`` is a *model wrapper* (not a ``LogitsProcessor``), so it can
legitimately host a training-time loss method.  ``compute_repbend_loss`` below
is the faithful five-term RepBend objective — pure tensor arithmetic (norms,
cosines, KL), CPU-importable and unit-testable with no GPU.  A caller plugs it
into a LoRA fine-tuning loop.

The legacy activation-capture hook is preserved but is now **opt-in** via
``capture_hook=True`` and is explicitly documented as a diagnostic helper, NOT
the paper's mechanism.  Passing ``safe_directions`` no longer silently
registers an inert "steering" hook.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from safetune.core.repeng.repbend import (
        RepBendConfig as _CoreRepBendConfig,
        RepBendWrapper,
    )
    _IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    RepBendWrapper = None  # type: ignore[assignment]
    _CoreRepBendConfig = None  # type: ignore[assignment]
    _IMPORT_ERROR = _e


class RepBendModel:
    """RepBend representation-bending wrapper (training-time method).

    RepBend is a fine-tuning method: it has **no inference-time intervention**.
    This wrapper therefore exposes RepBend's actual training objective via
    :meth:`compute_repbend_loss`, to be optimised inside a LoRA fine-tuning
    loop.  ``generate`` / ``__call__`` pass straight through to the base model
    (a RepBend-trained model already bends representations through its weights;
    a non-RepBend model is simply unmodified — there is, faithfully, nothing to
    apply at inference).

    Parameters
    ----------
    model
        The base causal-LM.
    safe_directions
        Optional precomputed per-layer ``mean(safe) - mean(unsafe)``
        directions.  Stored for diagnostics only; RepBend's faithful loss
        does **not** use a fixed direction (it compares fine-tuned vs frozen
        activations), so these are not used by :meth:`compute_repbend_loss`.
    target_layers
        Hidden-layer indices the loss is measured on.  RepBend works best on
        mid-to-later layers (paper: layers ~20+).
    bending_strength
        Back-compat alias retained in the public signature.

    Optional keyword-only kwargs (new — defaults preserve old construction)
    ----------------------------------------------------------------------
    loss_alpha, loss_beta, loss_gamma, loss_epsilon, loss_eta
        Coefficients for the five RepBend loss terms (safe / unsafe / cosine /
        KL / safe-unsafe).  Defaults follow the repo's convention of using a
        non-trivial mix; ``loss_eta`` defaults to 0.0 since the safe-unsafe
        term needs paired data.
    kl_temperature
        Temperature for the KL retain term (repo default 2.0).
    capture_hook
        If ``True``, register the legacy diagnostic activation-capture hook.
        This is **not** RepBend's mechanism and applies nothing to outputs;
        it is off by default.
    """

    def __init__(
        self,
        model: Any,
        safe_directions: Optional[Dict[int, Any]] = None,
        target_layers: Optional[List[int]] = None,
        bending_strength: float = 0.3,
        *,
        loss_alpha: float = 1.0,
        loss_beta: float = 1.0,
        loss_gamma: float = 1.0,
        loss_epsilon: float = 1.0,
        loss_eta: float = 0.0,
        kl_temperature: float = 2.0,
        capture_hook: bool = False,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.repeng.repbend is unavailable"
            ) from _IMPORT_ERROR
        self.model = model

        cfg_kwargs: Dict[str, Any] = {"bending_strength": bending_strength}
        if target_layers is not None:
            cfg_kwargs["target_layers"] = list(target_layers)
        config = _CoreRepBendConfig(**cfg_kwargs)
        self._impl = RepBendWrapper(model=model, config=config)
        self.target_layers: List[int] = list(config.target_layers)

        # Loss coefficients (faithful five-term RepBend objective).
        self.loss_alpha = float(loss_alpha)
        self.loss_beta = float(loss_beta)
        self.loss_gamma = float(loss_gamma)
        self.loss_epsilon = float(loss_epsilon)
        self.loss_eta = float(loss_eta)
        self.kl_temperature = float(kl_temperature)
        self.bending_strength = float(bending_strength)

        # Stored for diagnostics only — NOT used by the faithful loss.
        self.safe_directions = safe_directions
        if safe_directions is not None:
            self._impl.set_directions(safe_directions)

        # The legacy capture hook is now opt-in and is a diagnostic, not the
        # paper's mechanism.  Passing safe_directions no longer auto-registers
        # an inert "steering" hook.
        self._capture_hook_active = False
        if capture_hook:
            self._impl.register_hooks()
            self._capture_hook_active = True

    # ── faithful RepBend training-time loss ──────────────────────────

    def compute_repbend_loss(
        self,
        safe_hidden: Dict[int, Any],
        orig_safe_hidden: Dict[int, Any],
        unsafe_hidden: Dict[int, Any],
        orig_unsafe_hidden: Dict[int, Any],
        retain_logits: Optional[Any] = None,
        orig_retain_logits: Optional[Any] = None,
        unsafe_response_hidden: Optional[Dict[int, Any]] = None,
        safe_response_target_hidden: Optional[Dict[int, Any]] = None,
    ) -> Any:
        """Faithful RepBend fine-tuning loss (arXiv:2504.01550, ``_calc_loss``).

        Implements the verbatim aggregation from the RepBend repo::

            loss = alpha * safe_loss
                 - beta  * unsafe_loss
                 + gamma * cosine_loss
                 + eps   * kl_loss
                 + eta   * safe_unsafe_loss

        All hidden-state arguments are dicts ``{layer_idx -> tensor}``; per the
        paper the layer index keys the residual stream at the output of that
        transformer block.  Tensors are ``(batch, seq, dim)`` (or already
        pooled to ``(batch, dim)``).  ``M'`` denotes the *fine-tuned* model
        currently being trained; ``orig_*`` denotes the *frozen original*
        model ``M``.

        Parameters
        ----------
        safe_hidden, orig_safe_hidden
            Fine-tuned / frozen hidden states on **safe** prompts.  Their
            difference is ``v_s``; ``safe_loss = mean ||v_s||_2`` keeps benign
            representations close to the original model.
        unsafe_hidden, orig_unsafe_hidden
            Fine-tuned / frozen hidden states on **unsafe** prompts.  Their
            difference is ``v_u``; ``unsafe_loss = mean ||v_u||_2`` is entered
            with a *negative* coefficient so it is **maximised** — bending
            harmful representations far from the original.
        retain_logits, orig_retain_logits
            Fine-tuned / frozen output logits on the **retain** (benign
            instruction-following) set.  Yields the temperature-scaled KL term
            ``KL(M'||M)`` that preserves general capability.  Optional.
        unsafe_response_hidden, safe_response_target_hidden
            Fine-tuned hidden states over the *response* region of an unsafe
            prompt, and the frozen-model hidden states of the *safe* response
            to that same unsafe prompt.  Their L2 difference is the
            ``safe_unsafe_loss`` term.  Optional (needs ``loss_eta > 0``).

        Returns
        -------
        torch.Tensor
            Scalar differentiable loss.
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:  # pragma: no cover
            raise ImportError("RepBend requires PyTorch.")

        def _pool(t: Any) -> Any:
            # Reduce (batch, seq, dim) -> (batch, dim) by token-mean; leave
            # already-pooled (batch, dim) tensors untouched.
            t = t.float()
            return t.mean(dim=1) if t.dim() == 3 else t

        def _common_layers(a: Dict[int, Any], b: Dict[int, Any]) -> List[int]:
            return [l for l in self.target_layers if l in a and l in b] or [
                l for l in sorted(set(a) & set(b))
            ]

        device = None
        for d in (safe_hidden, orig_safe_hidden, unsafe_hidden, orig_unsafe_hidden):
            for v in d.values():
                device = v.device
                break
            if device is not None:
                break
        zero = torch.zeros((), dtype=torch.float, device=device)

        # ── safe_loss: mean ||v_s|| = ||M'(safe) - M(safe)|| ──────────
        safe_loss = zero
        s_layers = _common_layers(safe_hidden, orig_safe_hidden)
        if s_layers:
            acc = zero
            for l in s_layers:
                v_s = _pool(safe_hidden[l]) - _pool(orig_safe_hidden[l])
                acc = acc + v_s.norm(p=2, dim=-1).mean()
            safe_loss = acc / len(s_layers)

        # ── unsafe_loss: mean ||v_u|| = ||M'(unsafe) - M(unsafe)|| ────
        # entered NEGATIVELY below -> maximised (push reps far away).
        unsafe_loss = zero
        u_layers = _common_layers(unsafe_hidden, orig_unsafe_hidden)
        if u_layers:
            acc = zero
            for l in u_layers:
                v_u = _pool(unsafe_hidden[l]) - _pool(orig_unsafe_hidden[l])
                acc = acc + v_u.norm(p=2, dim=-1).mean()
            unsafe_loss = acc / len(u_layers)

        # ── cosine_loss: 1 - mean pairwise cos_sim of unsafe reps ─────
        # Encourages harmful representations to collapse together so the
        # model consistently refuses them (repo: _calc_cosine_loss).
        cosine_loss = zero
        if self.loss_gamma > 0 and u_layers:
            acc = zero
            n = 0
            for l in u_layers:
                h = _pool(unsafe_hidden[l])              # (B, D)
                if h.shape[0] < 2:
                    continue
                hn = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
                sim = hn @ hn.t()                        # (B, B)
                eye = torch.eye(h.shape[0], device=h.device, dtype=torch.bool)
                acc = acc + (1.0 - sim[~eye]).mean()
                n += 1
            if n > 0:
                cosine_loss = acc / n

        # ── kl_loss: temperature-scaled KL( M'(retain) || M(retain) ) ─
        kl_loss = zero
        if (
            self.loss_epsilon > 0
            and retain_logits is not None
            and orig_retain_logits is not None
        ):
            temp = self.kl_temperature
            p = F.log_softmax(retain_logits.float() / temp, dim=-1)
            q = F.softmax(orig_retain_logits.float() / temp, dim=-1)
            kl_loss = F.kl_div(p, q, reduction="batchmean") * (temp ** 2)

        # ── safe_unsafe_loss: bend unsafe-prompt response toward safe ─
        safe_unsafe_loss = zero
        if (
            self.loss_eta > 0
            and unsafe_response_hidden is not None
            and safe_response_target_hidden is not None
        ):
            su_layers = _common_layers(
                unsafe_response_hidden, safe_response_target_hidden
            )
            if su_layers:
                acc = zero
                for l in su_layers:
                    diff = (
                        unsafe_response_hidden[l].float()
                        - safe_response_target_hidden[l].float()
                    )
                    acc = acc + diff.norm(p=2, dim=-1).mean()
                safe_unsafe_loss = acc / len(su_layers)

        # ── verbatim aggregation (RepBend repo, _calc_loss) ───────────
        loss = (
            self.loss_alpha * safe_loss
            - self.loss_beta * unsafe_loss
            + self.loss_gamma * cosine_loss
            + self.loss_epsilon * kl_loss
            + self.loss_eta * safe_unsafe_loss
        )
        return loss

    # ── pass-through generation (RepBend applies nothing at inference) ──

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Pass through to the base model.

        RepBend has no inference-time intervention: a RepBend-trained model
        already bends representations via its weights; this wrapper does not
        and cannot add a faithful runtime edit.
        """
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        return getattr(self.__dict__["model"], name)

    def remove_hooks(self) -> None:
        if hasattr(self._impl, "remove_hooks"):
            self._impl.remove_hooks()
        self._capture_hook_active = False

    def __enter__(self) -> "RepBendModel":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "RepBendModel":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(path)
        return cls(model, **kwargs)
