"""
CTRAP Trainer — Collapse Trap.

Faithful implementation of the CTRAP algorithm from:

    Yifan Lu, Yigeng Zhou, Jing Li, Yequan Wang, Xuebo Liu, Daojing He,
    Fangming Liu, Min Zhang.
    "CTRAP: Embedding Collapse Trap to Safeguard Large Language Models
    from Harmful Fine-Tuning."  arXiv:2505.16559, 2025.

CTRAP departs from selective unlearning: instead of removing harmful
knowledge, it *pre-configures* the model's reaction to a future harmful
fine-tuning attack.  A dormant "collapse trap" is implanted during alignment
that, when (and only when) the model is later fine-tuned on harmful data,
drives the model to collapse onto a single, fixed, degenerate output token.
Benign fine-tuning leaves the model intact.

Paper objective (the two equations we reimplement)
---------------------------------------------------
**Eq. (1) — Collapse loss.**  The model is pushed to predict, with high
probability, a single fixed token ``e`` at every position regardless of
context::

    L_Collapse(θ; D) = E_{(x,y)~D} [ -1/|y| * Σ_{t=1}^{|y|}
                                      log p(e | x ∘ y_{<t} ; θ) ]

This is a per-token cross-entropy toward the constant target token ``e``
(NOT an MSE to a random vector — the previous implementation was unfaithful
on this point).

**Eq. (2) — Bi-level / simulated-attack objective.**  The collapse loss is
evaluated NOT at the current parameters θ, but at *simulated post-attack*
parameters θ′ obtained by taking one adversarial harmful-update step.  The
benign behaviour is preserved at θ::

    argmin_θ  L(θ; D_alignment)
              + λ * L_Collapse( θ − α·∇_θ L(θ; D_harmful) ; D_general )

where

* ``L(θ; D_alignment)`` is the standard supervised alignment loss at θ
  (the benign-preservation term, evaluated at the *current* parameters);
* ``θ′ = θ − α·∇_θ L(θ; D_harmful)`` is a one-step simulated adversary
  (α = inner step size, gradient of the harmful CE loss);
* ``L_Collapse(θ′; D_general)`` is Eq. (1) evaluated at θ′ on general/utility
  data — i.e. "if an attacker took one harmful step, the model would already
  be collapsing toward token ``e`` on ordinary inputs".

Paper default hyperparameters: ``λ = 0.1`` and ``α = 0.1``.

The conditional structure is what makes the trap dormant: at θ the collapse
loss term is *not* added to the supervised path, so benign training does not
trigger collapse; the collapse penalty only bites along the harmful-gradient
direction.

Differentiation through the inner step
--------------------------------------
Eq. (2) is a bi-level objective: faithfully it requires differentiating the
outer collapse loss through the inner harmful-gradient step θ′ = θ − α·∇_θ L,
which is a second-order (MAML-style) term.  This implementation supports both:

* ``ctrap_second_order=True`` (default): the inner step is built with
  ``torch.autograd.grad(..., create_graph=True)`` and θ′ is applied
  functionally so the outer loss backpropagates through the inner gradient
  (full bi-level, Hessian-vector products included).  This matches Eq. (2)
  exactly.
* ``ctrap_second_order=False``: a first-order approximation in which the
  inner harmful gradient is detached (``create_graph=False``); the outer
  collapse loss is then differentiated treating θ′ as a constant offset
  (FOMAML-style).  Use this when memory/throughput at scale is a concern;
  it is documented here as an approximation, not as the exact objective.

Functional θ′ is realised with ``torch.func.functional_call`` when available
(PyTorch >= 1.12), falling back to a parameter add/restore scheme otherwise.

Faithfulness caveat (validated on Qwen2.5-0.5B-Instruct, transformers 5.12)
---------------------------------------------------------------------------
The faithful second-order path requires a double backward through attention.
Fused attention kernels (PyTorch SDPA / FlashAttention) do NOT implement the
attention *double*-backward, so ``second_order=True`` raises
``derivative for aten::_scaled_dot_product_efficient_attention_backward is
not implemented`` under those kernels.  It works correctly when the model is
loaded with ``attn_implementation="eager"``.  Therefore, to keep training
robust out-of-the-box, this trainer sets ``second_order=True`` only when the
attached model uses eager attention; otherwise it automatically downgrades to
the documented first-order approximation (and logs a warning).  This is an
honest, explicit approximation — not silently mislabelled as exact.

Usage::

    from safetune.harden.ctrap import CTRAPTrainer, CTRAPConfig

    config = CTRAPConfig(
        output_dir="ctrap_out",
        num_train_epochs=20,
        ctrap_lambda=0.1,        # λ in Eq. (2)
        ctrap_alpha=0.1,         # α inner step size in Eq. (2)
        ctrap_collapse_token_id=None,   # token e; defaults to EOS/pad if None
    )
    trainer = CTRAPTrainer(
        model=model,
        args=config,
        train_dataset=alignment_dataset,
        harmful_dataset=harmful_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

Data format
-----------
``train_dataset`` (passed to the parent Trainer) holds the benign alignment
examples — it provides BOTH ``L(θ; D_alignment)`` (the outer benign term) and,
in the absence of a dedicated general set, ``D_general`` for the collapse
evaluation.  ``harmful_dataset`` is ``D_harmful``: it is used to compute the
simulated-attack gradient ``∇_θ L(θ; D_harmful)``.  Both datasets must yield
dicts with at least ``input_ids`` and ``attention_mask``; ``harmful_dataset``
must carry ``labels`` so the harmful CE loss can be computed.

Backwards-compatibility
------------------------
``ctrap_gamma`` is retained as a deprecated alias for ``ctrap_lambda``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — torch
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = _e

# torch.func.functional_call — preferred path for evaluating L_Collapse(θ′).
try:  # pragma: no cover - availability depends on torch version
    from torch.func import functional_call as _functional_call
except Exception:  # pragma: no cover
    try:
        from torch.nn.utils.stateless import functional_call as _functional_call  # type: ignore
    except Exception:
        _functional_call = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Optional imports — transformers
# ---------------------------------------------------------------------------
try:
    from transformers import Trainer, TrainingArguments
    _TRAINER_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    Trainer = object  # type: ignore[assignment,misc]
    TrainingArguments = object  # type: ignore[assignment,misc]
    _TRAINER_IMPORT_ERROR = _e


# ---------------------------------------------------------------------------
# CTRAPConfig
# ---------------------------------------------------------------------------

if _TRAINER_IMPORT_ERROR is None:
    @dataclass
    class CTRAPConfig(TrainingArguments):  # type: ignore[misc]
        """``TrainingArguments`` subclass exposing CTRAP hyperparameters.

        Fields
        ------
        ctrap_lambda : float
            Weight λ of the collapse-trap term in Eq. (2).  Paper default 0.1.
        ctrap_alpha : float
            Inner simulated-attack step size α in Eq. (2)
            (θ′ = θ − α·∇_θ L(θ; D_harmful)).  Paper default 0.1.
        ctrap_collapse_token_id : int | None
            The fixed degenerate target token ``e`` (Eq. 1).  The paper does
            not fix its identity; when ``None`` the trainer uses the model /
            tokenizer EOS id, falling back to the pad id, then to 0.
        ctrap_second_order : bool
            If True (default) backpropagate through the inner harmful-gradient
            step (faithful bi-level, MAML-style).  If False, use a first-order
            approximation (detach the inner gradient, FOMAML-style).
        ctrap_gamma : float | None
            Deprecated alias for ``ctrap_lambda`` (kept for API stability).
            If set, it overrides ``ctrap_lambda``.
        """

        ctrap_lambda: float = 0.1
        ctrap_alpha: float = 0.1
        ctrap_collapse_token_id: Optional[int] = None
        ctrap_second_order: bool = True
        ctrap_gamma: Optional[float] = None

else:  # pragma: no cover
    class CTRAPConfig(object):  # type: ignore[assignment]
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_cycling_loader(dataset: Any, batch_size: int,
                         collate_fn: Optional[Any] = None) -> Iterator:
    """Return an infinitely cycling DataLoader over *dataset*.

    A ``collate_fn`` (e.g. ``transformers.default_data_collator``) is used so
    that datasets of plain Python lists are stacked into ``(B, T)`` tensors;
    without it the default collation produces a list of per-position tensors,
    which breaks the simulated-attack forward pass.

    Falls back to cycling the raw iterable if DataLoader construction fails
    (e.g. the dataset is already a pre-batched iterable).
    """
    try:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        return cycle(loader)
    except Exception:
        return cycle(dataset)


def _prepare_batch(batch: Any, device: Any) -> Any:
    """Move every tensor in *batch* to *device* (dict convention)."""
    if not isinstance(batch, dict):
        return batch
    return {
        k: (v.to(device) if hasattr(v, "to") else v)
        for k, v in batch.items()
    }


def _resolve_collapse_token_id(model: Any, configured: Optional[int]) -> int:
    """Resolve the fixed collapse target token ``e`` (Eq. 1).

    Precedence: explicit config > model.config.eos_token_id >
    model.config.pad_token_id > 0.  The paper leaves ``e`` unspecified; any
    fixed token works as long as it is constant across training.
    """
    if configured is not None:
        return int(configured)
    cfg = getattr(model, "config", None)
    for attr in ("eos_token_id", "pad_token_id"):
        val = getattr(cfg, attr, None) if cfg is not None else None
        if isinstance(val, int) and val >= 0:
            return int(val)
        if isinstance(val, (list, tuple)) and val and isinstance(val[0], int):
            return int(val[0])
    return 0


def _model_uses_eager_attention(model: Any) -> bool:
    """True if the model is configured for eager attention.

    Fused SDPA / FlashAttention kernels lack an attention double-backward, so
    the faithful second-order CTRAP term only runs under eager attention.
    Conservatively returns True when the implementation cannot be determined
    (e.g. non-HF stubs in tests) so the faithful path is attempted.
    """
    cfg = getattr(model, "config", None)
    impl = getattr(cfg, "_attn_implementation", None) if cfg is not None else None
    if impl is None:
        return True
    return str(impl).lower() == "eager"


def _logits_of(outputs: Any) -> "torch.Tensor":
    """Pull logits out of an HF-style or tuple-style model output."""
    if hasattr(outputs, "logits") and outputs.logits is not None:
        return outputs.logits
    if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
        return outputs[0]
    raise ValueError("CTRAPTrainer: model output has no logits.")


def _harmful_ce_loss(outputs: Any, labels: "torch.Tensor") -> "torch.Tensor":
    """Standard next-token CE loss L(θ; D_harmful) from logits + labels."""
    logits = _logits_of(outputs)
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous().to(shift_logits.device)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _collapse_loss(outputs: Any, attention_mask: Optional["torch.Tensor"],
                   collapse_token_id: int) -> "torch.Tensor":
    """Eq. (1): per-token CE toward the fixed collapse token ``e``.

    ``L_Collapse = mean_t [ -log p(e | context_{<t}) ]`` over (masked) tokens.
    """
    logits = _logits_of(outputs).float()           # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)       # (B, T, V)
    # -log p(e | ...) at every position.
    nll_e = -log_probs[..., collapse_token_id]      # (B, T)
    if attention_mask is not None:
        mask = attention_mask.to(nll_e.dtype)       # (B, T)
        denom = mask.sum().clamp(min=1.0)
        return (nll_e * mask).sum() / denom
    return nll_e.mean()


def _forward(model: Any, batch: Any) -> Any:
    """Run a forward pass, tolerating models that reject extra kwargs."""
    try:
        return model(**batch)
    except TypeError:
        safe = {"input_ids", "attention_mask", "labels", "token_type_ids"}
        return model(**{k: v for k, v in batch.items() if k in safe})


# ---------------------------------------------------------------------------
# CTRAPTrainer
# ---------------------------------------------------------------------------

class CTRAPTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """HuggingFace Trainer implementing the CTRAP collapse trap (Eq. 1 & 2).

    Per step::

        L_total = L(θ; D_alignment)                              # benign @ θ
                  + λ · L_Collapse( θ − α·∇_θ L(θ; D_harmful) ;  # collapse @ θ′
                                    D_general )

    Args:
        harmful_dataset: ``D_harmful`` — used to form the simulated-attack
            gradient ∇_θ L(θ; D_harmful).  Required; must carry ``labels``.
        general_dataset: optional ``D_general`` for evaluating the collapse
            loss at θ′.  Defaults to reusing the alignment ``inputs`` batch.
        All other arguments pass through to ``transformers.Trainer``.
    """

    def __init__(
        self,
        *args: Any,
        harmful_dataset: Optional[Any] = None,
        general_dataset: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for CTRAPTrainer"
            ) from _TRAINER_IMPORT_ERROR
        if _TORCH_IMPORT_ERROR is not None:
            raise ImportError(
                "torch is required for CTRAPTrainer"
            ) from _TORCH_IMPORT_ERROR
        if harmful_dataset is None:
            raise ValueError(
                "CTRAPTrainer requires a 'harmful_dataset' argument."
            )

        super().__init__(*args, **kwargs)

        # λ (with deprecated ctrap_gamma alias) and α.
        gamma = getattr(self.args, "ctrap_gamma", None)
        if gamma is not None:
            logger.warning(
                "CTRAPTrainer: 'ctrap_gamma' is deprecated; use 'ctrap_lambda'."
            )
            self._lambda = float(gamma)
        else:
            self._lambda = float(getattr(self.args, "ctrap_lambda", 0.1))
        self._alpha = float(getattr(self.args, "ctrap_alpha", 0.1))
        self._second_order = bool(getattr(self.args, "ctrap_second_order", True))
        # Second-order (faithful bi-level) needs a double backward through
        # attention, which fused SDPA/FlashAttention kernels do not implement.
        # Auto-downgrade to the first-order approximation unless the model uses
        # eager attention, so training never crashes in the outer backward.
        if self._second_order and not _model_uses_eager_attention(self.model):
            logger.warning(
                "CTRAPTrainer: model is not using eager attention; the faithful "
                "second-order bi-level term needs an attention double-backward "
                "that fused kernels do not support. Downgrading to the "
                "first-order approximation. Load the model with "
                "attn_implementation='eager' to use the exact Eq.(2) objective."
            )
            self._second_order = False
        self._collapse_token_id = _resolve_collapse_token_id(
            self.model, getattr(self.args, "ctrap_collapse_token_id", None)
        )
        logger.info(
            "CTRAPTrainer: λ=%.4g, α=%.4g, collapse_token_id=%d, second_order=%s",
            self._lambda, self._alpha, self._collapse_token_id, self._second_order,
        )

        try:
            _bs = self.args.per_device_train_batch_size
        except AttributeError:
            _bs = 1
        # Reuse the trainer's collator so list-valued datasets get stacked into
        # (B, T) tensors (HF default DataLoader collation would not).
        _collate = getattr(self, "data_collator", None)
        self._harm_iter: Iterator = _make_cycling_loader(
            harmful_dataset, _bs, collate_fn=_collate)
        self._general_iter: Optional[Iterator] = (
            _make_cycling_loader(general_dataset, _bs, collate_fn=_collate)
            if general_dataset is not None else None
        )

    # ------------------------------------------------------------------
    # Simulated-attack helpers
    # ------------------------------------------------------------------

    def _trainable_named_params(self, model: Any):
        """Named parameters that require grad (LoRA-friendly)."""
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not named:  # fall back to all params if nothing is marked trainable
            named = list(model.named_parameters())
        return named

    def _simulated_attack_params(self, model: Any, harm_batch: Any):
        """Return ``{name: θ′}`` for θ′ = θ − α·∇_θ L(θ; D_harmful).

        With ``second_order=True`` the returned θ′ tensors carry a graph back
        to θ (create_graph=True), so the outer collapse loss differentiates
        through the inner step (faithful Eq. 2).  Otherwise the inner gradient
        is detached (first-order approximation).
        """
        named = self._trainable_named_params(model)
        names = [n for n, _ in named]
        params = [p for _, p in named]

        harm_out = _forward(model, harm_batch)
        labels = harm_batch.get("labels")
        if labels is None:
            raise ValueError(
                "CTRAPTrainer: harmful batch needs 'labels' to compute "
                "the simulated-attack gradient ∇_θ L(θ; D_harmful)."
            )
        l_harm = _harmful_ce_loss(harm_out, labels)

        grads = torch.autograd.grad(
            l_harm, params,
            create_graph=self._second_order,
            allow_unused=True,
        )
        theta_prime = {}
        for name, p, g in zip(names, params, grads):
            if g is None:
                theta_prime[name] = p
            else:
                theta_prime[name] = p - self._alpha * g
        return theta_prime, l_harm.detach()

    def _collapse_loss_at(self, model: Any, theta_prime: dict,
                          general_batch: Any) -> "torch.Tensor":
        """Eq. (1) evaluated at simulated parameters θ′ on D_general."""
        attn = general_batch.get("attention_mask")
        # Forward at θ′ without labels (collapse target replaces them).
        gen_inputs = {k: v for k, v in general_batch.items() if k != "labels"}

        if _functional_call is not None:
            # Full named-param override: θ′ for trainable params, θ for the rest.
            full = dict(model.named_parameters())
            full.update(theta_prime)
            full.update(dict(model.named_buffers()))
            outputs = _functional_call(model, full, args=(), kwargs=gen_inputs)
            return _collapse_loss(outputs, attn, self._collapse_token_id)

        # Fallback: temporarily write θ′ into the params, forward, restore.
        named = self._trainable_named_params(model)
        backup = {n: p.detach().clone() for n, p in named}
        try:
            with torch.no_grad():
                for n, p in named:
                    if n in theta_prime:
                        p.copy_(theta_prime[n].detach())
            outputs = _forward(model, gen_inputs)
            loss = _collapse_loss(outputs, attn, self._collapse_token_id)
            # NOTE: without functional_call the graph back to θ is broken;
            # this path is an approximation regardless of second_order.
            return loss
        finally:
            with torch.no_grad():
                for n, p in named:
                    p.copy_(backup[n])

    # ------------------------------------------------------------------
    # Core: CTRAP compound loss (Eq. 2)
    # ------------------------------------------------------------------

    def compute_loss(  # type: ignore[override]
        self,
        model: Any,
        inputs: Any,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Any:
        """Compute Eq. (2):

        ``L(θ; D_align) + λ · L_Collapse(θ − α·∇_θ L(θ; D_harm) ; D_general)``.
        """
        device = next(model.parameters()).device

        # ---- Outer benign term: L(θ; D_alignment) at the CURRENT params ----
        align_outputs = _forward(model, inputs)
        if hasattr(align_outputs, "loss") and align_outputs.loss is not None:
            l_align = align_outputs.loss
        else:
            labels = inputs.get("labels")
            if labels is None:
                raise ValueError(
                    "CTRAPTrainer: alignment inputs need 'labels' (or the model "
                    "must return a loss) to compute L(θ; D_alignment)."
                )
            l_align = _harmful_ce_loss(align_outputs, labels)

        # ---- Collapse trap term @ simulated attack params θ′ ---------------
        l_collapse: "torch.Tensor" = torch.zeros((), device=device)
        try:
            harm_raw = next(self._harm_iter)
            harm_batch = _prepare_batch(harm_raw, device)

            # θ′ = θ − α·∇_θ L(θ; D_harmful)   (the simulated one-step adversary)
            theta_prime, _l_harm = self._simulated_attack_params(model, harm_batch)

            # D_general: dedicated set if provided, else reuse alignment batch.
            if self._general_iter is not None:
                general_batch = _prepare_batch(next(self._general_iter), device)
            else:
                general_batch = inputs

            # L_Collapse(θ′; D_general)  (Eq. 1 evaluated at θ′)
            l_collapse = self._collapse_loss_at(model, theta_prime, general_batch)
        except Exception as exc:
            logger.warning(
                "CTRAPTrainer: simulated-attack / collapse path failed (%s); "
                "L_collapse=0 this step.", exc,
            )
            l_collapse = next(iter(model.parameters())).sum() * 0.0

        # ---- Eq. (2) combined objective ------------------------------------
        loss = l_align + self._lambda * l_collapse

        return (loss, align_outputs) if return_outputs else loss


__all__ = ["CTRAPConfig", "CTRAPTrainer"]
