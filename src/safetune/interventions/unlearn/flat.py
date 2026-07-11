"""
FLAT: LLM Unlearning via Loss Adjustment with Only Forget Data.

Reference: Wang, Wei, Liu, Pang, Liu, Shah, Bao, Liu, Wei,
"LLM Unlearning via Loss Adjustment with Only Forget Data", ICLR 2025
(arXiv:2410.11143).  Official code: https://github.com/UCSC-REAL/FLAT.

FLAT = "Forget data only Loss AjustmenT".  The method needs *only* the forget
data: for each forget prompt it pairs the response-to-be-forgotten with a
template/exemplar "good" answer (in the safety setting, a refusal).  It then
adjusts the loss by **maximising an f-divergence** between the template-answer
distribution and the forget-answer distribution, using the variational
(f-GAN) form of the divergence::

    D_f(P_good ‖ P_forget) ≥ E_good[ g(T) ] − E_forget[ f*(g(T)) ]

where ``T`` is realised through the model's per-answer log-likelihood, ``g`` is
the divergence's output activation and ``f*`` is the convex conjugate.  Training
*minimises* the negated lower bound, which (i) pushes the **template/good**
answer's likelihood **up** and (ii) pushes the **forget** answer's likelihood
**down** — with the per-term weighting set by the chosen divergence rather than
a hand-tuned coefficient.  Unlike DPO/NPO, FLAT is reference-free: no frozen
oracle model is required.

Supported f-divergences (``FLATConfig.divergence``): ``kl``, ``reverse_kl``,
``jeffrey``, ``squared_hellinger``, ``pearson``, ``neyman``,
``jensen_shannon``, ``total_variation`` — matching the official release.

For safety unlearning:
  * the **forget** answer = the harmful completion to a harmful prompt;
  * the **good/template** answer = a safe refusal to the same prompt;
  * an optional **retain** CE term (``flat_retain``) on benign data anchors
    general capability, mirroring the GradDiff retain branch.

NOTE ON SCALE: the official code sums the per-answer log-likelihood; we use the
*mean per answer token* (a per-sequence average NLL) instead, which keeps the
divergence arguments in an O(1) range so the ``exp``/``sqrt`` conjugates stay
numerically stable across answer lengths.  The optimisation direction is
identical.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_VARIANTS = ("flat", "flat_retain")

# Canonical divergence names (official code labels mapped to snake_case keys).
_DIVERGENCES = (
    "kl",
    "reverse_kl",
    "jeffrey",
    "squared_hellinger",
    "pearson",
    "neyman",
    "jensen_shannon",
    "total_variation",
)


@dataclass
class FLATConfig:
    """Configuration for FLAT unlearning.

    Attributes:
        variant: ``"flat"`` — f-divergence forget/template term only;
            ``"flat_retain"`` — adds a CE retain term on benign data to anchor
            capability (recommended).
        divergence: which f-divergence to maximise between the template-answer
            and forget-answer distributions.  One of :data:`_DIVERGENCES`
            (the full menu from the official release).  ``"kl"`` (the default),
            ``"jensen_shannon"``, ``"total_variation"`` and ``"jeffrey"`` are
            the validated, directionally-stable choices (template likelihood
            up, forget likelihood down across operating regimes).  The
            remaining conjugates (``reverse_kl``, ``pearson``, ``neyman``,
            ``squared_hellinger``) reproduce the paper's menu but have
            restricted valid regimes and can change sign for some
            answer-length / likelihood ranges — prefer the validated set.
        epochs: passes over the (forget, good[, retain]) iterables.
        lr: AdamW learning rate.  ``1e-5`` matches TOFU / NPO defaults.
        weight_decay: AdamW weight decay.  ``0.01`` matches TOFU defaults.
        retain_coeff: weight on the retain CE term for ``flat_retain``.
        forget_clip: optional max global grad-norm clip applied before each
            optimizer step (``None`` disables clipping).  Replaces the old
            loss-magnitude clip; same config knob name for compatibility.
        max_steps: optional hard cap on total optimizer steps.
    """

    variant: str = "flat_retain"
    divergence: str = "kl"
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 0.01
    retain_coeff: float = 1.0
    forget_clip: Optional[float] = None
    max_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.variant not in _VARIANTS:
            raise ValueError(
                f"FLATConfig.variant must be one of {_VARIANTS}, "
                f"got {self.variant!r}"
            )
        self.divergence = _normalize_divergence(self.divergence)


def _normalize_divergence(name: str) -> str:
    """Map a user-supplied divergence label to a canonical key."""
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "rkl": "reverse_kl",
        "js": "jensen_shannon",
        "jenson_shannon": "jensen_shannon",  # official code's spelling
        "tv": "total_variation",
        "hellinger": "squared_hellinger",
    }
    key = aliases.get(key, key)
    if key not in _DIVERGENCES:
        raise ValueError(
            f"FLATConfig.divergence must be one of {_DIVERGENCES} "
            f"(got {name!r})"
        )
    return key


# ── f-divergence activation g(·) and convex conjugate f*(·) ─────────────────────
#
# Faithful to the official FLAT release (get_contrastive_loss).  Each pair maps
# a per-example tensor to a scalar.  Numerically-sensitive branches use softplus
# instead of ``log(1+exp(-x))`` for stability; the closed forms are unchanged.

def _g_fstar(divergence: str) -> Tuple[Callable, Callable]:
    if divergence == "kl":
        g = lambda x: -torch.mean(x)
        fstar = lambda x: -torch.mean(torch.exp(x - 1.0))
    elif divergence == "reverse_kl":
        g = lambda x: -torch.mean(-torch.exp(x))
        fstar = lambda x: -torch.mean(-1.0 - x)
    elif divergence == "jeffrey":
        g = lambda x: -torch.mean(x)
        fstar = lambda x: -torch.mean(x + x * x / 4.0 + x * x * x / 16.0)
    elif divergence == "squared_hellinger":
        g = lambda x: -torch.mean(1.0 - torch.exp(x))
        fstar = lambda x: -torch.mean((1.0 - torch.exp(x)) / torch.exp(x))
    elif divergence == "pearson":
        g = lambda x: -torch.mean(x)
        fstar = lambda x: -torch.mean(x * x / 4.0 + x)
    elif divergence == "neyman":
        g = lambda x: -torch.mean(1.0 - torch.exp(x))
        # 1 - x ≥ 1 here (x ≤ 0), so the sqrt domain is safe.
        fstar = lambda x: -torch.mean(2.0 - 2.0 * torch.sqrt(1.0 - x))
    elif divergence == "jensen_shannon":
        _log2 = torch.log(torch.tensor(2.0))
        # -log(1+exp(-x)) = -softplus(-x); x + log(1+exp(-x)) = x + softplus(-x)
        g = lambda x: -torch.mean(-F.softplus(-x)) - _log2.to(x.device)
        fstar = lambda x: -torch.mean(x + F.softplus(-x)) + _log2.to(x.device)
    elif divergence == "total_variation":
        g = lambda x: -torch.mean(torch.tanh(x) / 2.0)
        fstar = lambda x: -torch.mean(torch.tanh(x) / 2.0)
    else:  # pragma: no cover - guarded by _normalize_divergence
        raise ValueError(f"Unsupported f-divergence: {divergence!r}")
    return g, fstar


def _seq_mean_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean negative log-likelihood over the answer tokens.

    Args:
        logits: ``(batch, seq_len, vocab)`` model output logits.
        labels: ``(batch, seq_len)`` labels with ``-100`` on masked (prompt /
            pad) positions — the standard HF causal-LM convention.

    Returns:
        ``(batch,)`` tensor of mean per-token NLL over each sequence's valid
        (answer) tokens.  Sequences with no valid token contribute ``0``.
    """
    shift_logits = logits[..., :-1, :].contiguous()      # (B, L-1, V)
    shift_labels = labels[..., 1:].contiguous()          # (B, L-1)
    mask = (shift_labels != -100).to(shift_logits.dtype)  # (B, L-1)

    logp = F.log_softmax(shift_logits, dim=-1)
    gather_idx = shift_labels.clamp(min=0).unsqueeze(-1)  # (B, L-1, 1)
    tok_logp = logp.gather(-1, gather_idx).squeeze(-1)    # (B, L-1)
    nll_tok = -tok_logp * mask                            # zero on masked

    denom = mask.sum(dim=-1).clamp(min=1.0)               # (B,)
    return nll_tok.sum(dim=-1) / denom                    # (B,)


def flat_fdiv_loss(
    good_logits: torch.Tensor,
    good_labels: torch.Tensor,
    forget_logits: torch.Tensor,
    forget_labels: torch.Tensor,
    divergence: str = "kl",
) -> torch.Tensor:
    """FLAT f-divergence loss-adjustment objective (to be *minimised*).

    Maximises the variational lower bound of ``D_f(P_good ‖ P_forget)``;
    minimising the returned value therefore drives the **good/template**
    answer's likelihood up and the **forget** answer's likelihood down.

    Args:
        good_logits / good_labels: logits + labels for the template (good)
            answer (e.g. a safe refusal), HF convention (``-100`` masks).
        forget_logits / forget_labels: logits + labels for the forget answer
            (e.g. the harmful completion).
        divergence: f-divergence key (see :data:`_DIVERGENCES`).

    Returns:
        Scalar loss ready for ``.backward()``.
    """
    divergence = _normalize_divergence(divergence)
    g, fstar = _g_fstar(divergence)

    nll_good = _seq_mean_nll(good_logits, good_labels)        # (B,)
    nll_forget = _seq_mean_nll(forget_logits, forget_labels)  # (B,)

    # Official convention: feed the negated NLL into g/f*.
    prob_reg = -nll_good      # template-answer term
    prob_peer = -nll_forget   # forget-answer term

    loss_regular = g(prob_reg)
    loss_peer = fstar(prob_peer)
    return loss_regular - loss_peer


def _ce_retain_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard causal-LM cross-entropy on the retain set."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def _build_optimizer(model: nn.Module, cfg: FLATConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def flat_unlearn(
    model: nn.Module,
    forget_batches: Iterable[Dict[str, torch.Tensor]],
    good_batches: Iterable[Dict[str, torch.Tensor]],
    retain_batches: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
    *,
    forward_fn: Optional[
        Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor]
    ] = None,
    config: Optional[FLATConfig] = None,
) -> nn.Module:
    """Run FLAT unlearning in place on ``model``.

    No frozen reference model is required — FLAT is reference-free.

    Args:
        model: the model to unlearn.  Updated in place.
        forget_batches: iterable of forget-answer batches (harmful completions).
            Each batch needs ``input_ids``, ``attention_mask`` and ``labels``
            (``-100`` on prompt/pad positions).  Re-iterated once per epoch.
        good_batches: iterable of template/good-answer batches (e.g. safe
            refusals to the *same* prompts), aligned one-to-one with
            ``forget_batches``.  Same field convention.
        retain_batches: optional benign-data batches for the ``flat_retain``
            CE anchor; consumed one-per-forget-batch.
        forward_fn: optional ``(model, batch) -> logits`` callable.  If
            ``None``, ``model(**batch_without_labels).logits`` is used.
        config: :class:`FLATConfig`.

    Returns:
        The updated model (same object, mutated in place).
    """
    cfg = config or FLATConfig()
    if good_batches is None:
        raise ValueError(
            "flat_unlearn requires `good_batches` — FLAT pairs each forget "
            "answer with a template/good (e.g. refusal) answer for the same "
            "prompt. Build them e.g. via runner.unlearn.make_simdpo_pairs "
            "(chosen=good, rejected=forget)."
        )
    needs_retain = cfg.variant == "flat_retain"
    if needs_retain and retain_batches is None:
        raise ValueError("FLATConfig.variant='flat_retain' requires retain_batches.")

    def _logits(m: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if forward_fn is not None:
            return forward_fn(m, batch)
        out = m(**{k: v for k, v in batch.items() if k != "labels"})
        return out.logits if hasattr(out, "logits") else out

    opt = _build_optimizer(model, cfg)
    total_steps = 0
    max_steps = cfg.max_steps

    for epoch in range(1, max(1, cfg.epochs) + 1):
        if max_steps is not None and total_steps >= max_steps:
            break
        steps = 0
        good_iter = iter(good_batches)
        retain_iter = iter(retain_batches) if needs_retain else None

        for f_batch in forget_batches:
            if max_steps is not None and total_steps >= max_steps:
                break
            try:
                g_batch = next(good_iter)
            except StopIteration:
                logger.warning(
                    "flat_unlearn: good (template) iterable exhausted mid-epoch "
                    "%d; ending epoch early.", epoch,
                )
                break

            opt.zero_grad(set_to_none=True)

            # --- FLAT f-divergence loss adjustment --------------------------
            f_logits = _logits(model, f_batch)
            g_logits = _logits(model, g_batch)
            loss = flat_fdiv_loss(
                g_logits, g_batch["labels"],
                f_logits, f_batch["labels"],
                divergence=cfg.divergence,
            )

            # --- Retain CE term (flat_retain only) --------------------------
            if needs_retain:
                try:
                    r_batch = next(retain_iter)  # type: ignore[arg-type]
                except StopIteration:
                    logger.warning(
                        "flat_unlearn: retain iterable exhausted mid-epoch %d; "
                        "ending epoch early.", epoch,
                    )
                    break
                r_logits = _logits(model, r_batch)
                retain_loss = _ce_retain_loss(r_logits, r_batch["labels"])
                loss = loss + cfg.retain_coeff * retain_loss

            loss.backward()
            if cfg.forget_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=abs(cfg.forget_clip),
                )
            opt.step()
            steps += 1
            total_steps += 1

        logger.info(
            "flat_unlearn: epoch %d/%d (%s, div=%s) — %d steps.",
            epoch, cfg.epochs, cfg.variant, cfg.divergence, steps,
        )

    logger.info(
        "flat_unlearn: completed (%d total optimizer steps, variant=%s, div=%s).",
        total_steps, cfg.variant, cfg.divergence,
    )
    return model


__all__ = ["FLATConfig", "flat_unlearn", "flat_fdiv_loss"]
