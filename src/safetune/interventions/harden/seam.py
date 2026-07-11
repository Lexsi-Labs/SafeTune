"""
SEAM Trainer — Self-Destructive Language Models.

Faithful re-implementation of the SEAM objective from:

    Wang, Yang et al. "Self-Destructive Language Models."
    arXiv:2505.12186 (ICLR 2026).  Official code:
    https://github.com/ZJUWYH/seam

SEAM ("SElf-destructive lAnguage Model") trains a model so that legitimate
fine-tuning is preserved, but *harmful* fine-tuning drives the model into a
self-destructive optimisation trap that also wrecks its benign capabilities.
This is achieved by *coupling* the harmful-data gradient and the benign-data
gradient so they point in opposing directions.

Paper objective (Eq. 5)::

    L(theta) = L_ul(theta) + alpha * L_up(theta) + beta * L_sd(theta)

where (paper Eqs. 1-4):

* ``L_ul`` — *unlearning* loss on the adversarial set ``D_adv`` (Eq. 3).
  The official code (``src/core/trainer.py``, ``compute_loss(split="harmful")``)
  implements this as a layer-wise gradient-ascent loss adapted from RepNoise:
  the masked next-token cross-entropy is read out from *every* hidden state
  (projected through the final norm + LM head), averaged, and the objective
  is ``-log(mean_layer_ce + 1)`` so that descending it *raises* the harmful CE.

* ``L_up`` — *utility-preservation* loss on the alignment set ``D_aln``
  (Eq. 4): standard masked next-token cross-entropy.  ``D_aln`` is the set of
  harmful prompts paired with *refusal* responses (GPT-4o generated in the
  paper).  Weighted by ``alpha``.

* ``L_sd`` — the *self-destructive* gradient-coupling term (Eq. 2), the
  defining SEAM trap::

        L_sd(theta) = sim(g_a(theta), g_b(theta))
        g_a(theta)  = E_{(x,y)~D_adv} grad_theta ell(f_theta(x), y)    # Eq. 1
        g_b(theta)  = E_{(x,y)~D_bgn} grad_theta ell(f_theta(x), y)    # Eq. 1

  ``sim`` is the cosine similarity.  Minimising ``L_sd`` pushes the harmful
  fine-tuning gradient ``g_a`` and the benign gradient ``g_b`` to oppose each
  other, so an attacker descending ``g_a`` simultaneously ascends ``g_b``.
  Weighted by ``beta``.

Hessian-free estimator (paper Eq. 6, Theorem 1).  Directly differentiating the
cosine similarity ``L_sd`` w.r.t. ``theta`` needs the Hessian, which is
intractable for LLMs.  SEAM instead uses a first-order finite-difference
estimate of ``grad_theta L_sd``::

    gbar_a = g_a / ||g_a||,   gbar_b = g_b / ||g_b||,   c = gbar_a . gbar_b

    grad_theta L_sd_hat =
        (1/eps) * [ (g_b(theta + eps*(gbar_a - c*gbar_b)) - g_b(theta)) / ||g_b||
                  + (g_a(theta + eps*(gbar_b - c*gbar_a)) - g_a(theta)) / ||g_a|| ]

with perturbation radius ``eps << 1`` (paper default ``eps = 1e-3``).  The
estimate requires only forward/backward passes at perturbed parameters — no
Hessian.  Theorem 1 bounds the error by ``O(eps)``.

This module assembles the *exact* SEAM total gradient (paper Eq. 5 with the
Eq. 6 estimator for the ``L_sd`` part) by overriding ``training_step``:

    grad = g_ul + alpha * g_up + (beta/eps) * [finite-difference L_sd terms]

Differences from the official repo (documented for honesty)
-----------------------------------------------------------
* The official implementation shards the six forward/backward passes across
  three GPUs and accumulates gradients in a side dict via a custom
  ``_inner_training_loop``.  Here everything runs in a single ``training_step``
  on one device, which is correct for the 0.5B-scale models this package
  targets but uses more peak memory per step.
* The official ``L_ul`` / ``L_up`` use ``mask = (harmful_ids != harmless_ids)``
  to isolate response tokens from prompt-aligned harmful/harmless *pairs*.
  Our datasets follow the standard HF ``-100`` label convention instead, so we
  compute masked CE via ``ignore_index=-100`` (functionally the same masked
  next-token CE on response tokens).  The RepNoise layer-wise log term in
  ``L_ul`` is reproduced faithfully.
* If no separate ``benign_dataset`` / ``alignment_dataset`` are supplied, the
  benign batch falls back to the standard ``train_dataset`` batch and the
  alignment batch falls back to the harmful (refusal) set, so the public
  single-dataset API keeps working.

Usage::

    from safetune.harden.seam import SEAMTrainer, SEAMConfig

    config = SEAMConfig(output_dir="seam_out", num_train_epochs=3,
                        seam_alpha=1.0, seam_beta=0.001, seam_epsilon=1e-3)
    trainer = SEAMTrainer(
        model=model, args=config,
        train_dataset=benign_dataset,        # D_bgn (also the SFT stream)
        harmful_dataset=adversarial_dataset, # D_adv
        alignment_dataset=refusal_dataset,   # D_aln (optional)
        benign_dataset=benign_dataset,       # D_bgn (optional, else train_ds)
    )
    trainer.train()

Data format
-----------
Every dataset yields dicts with ``input_ids``, ``attention_mask`` and
``labels`` (standard HF collation).  Labels use the ``-100`` convention:
prompt / pad tokens are ``-100``; supervised response tokens carry real ids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — torch
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = _e

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
# SEAMConfig
# ---------------------------------------------------------------------------

if _TRAINER_IMPORT_ERROR is None:
    @dataclass
    class SEAMConfig(TrainingArguments):  # type: ignore[misc]
        """``TrainingArguments`` subclass exposing SEAM hyperparameters.

        Fields follow the paper (arXiv:2505.12186, Eq. 5) and the official
        repo defaults (``ZJUWYH/seam``: ``src/train.py``, ``config/train.yaml``).

        seam_alpha : float
            Weight of the utility-preservation term ``L_up`` (paper ``alpha``).
            Default ``1.0`` (official default).
        seam_beta : float
            Weight of the self-destructive gradient-coupling term ``L_sd``
            (paper ``beta``).  Default ``0.001`` (official ``src/train.py``
            default; the bundled ``config/train.yaml`` uses ``0.01``).
        seam_epsilon : float
            Perturbation radius ``eps`` for the Hessian-free estimator of
            ``grad L_sd`` (paper Eq. 6).  Default ``1e-3`` (official default).
        seam_refresh_every : int
            How often (in steps) to draw fresh harmful/alignment/benign
            batches.  Default ``1`` (every step).

        Backward-compat: ``seam_lambda`` is retained as a deprecated alias.
        When set to a non-default value it overrides ``seam_beta`` (the old
        naive ``L_align - lambda*L_harm`` objective scaled the harmful term;
        the closest faithful analogue is the coupling weight).
        """

        seam_alpha: float = 1.0
        seam_beta: float = 0.001
        seam_epsilon: float = 1e-3
        seam_refresh_every: int = 1

        # Deprecated alias from the previous (non-faithful) implementation.
        seam_lambda: float = 0.5

else:  # pragma: no cover
    class SEAMConfig(object):  # type: ignore[assignment]
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_cycling_loader(dataset: Any, batch_size: int, collate_fn=None) -> Iterator:
    """Return an infinitely cycling DataLoader over *dataset*."""
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


def _ce_loss(model: Any, batch: Any) -> Any:
    """Standard masked next-token CE (``ignore_index=-100``).

    Used for ``L_up`` (alignment) and the benign / harmful-attack CE that feed
    the gradient-coupling term.  Mirrors the official ``masked_token_ce_loss`` /
    ``ce_loss`` but using the HF ``-100`` label convention rather than the
    prompt-difference mask.
    """
    import torch.nn.functional as F
    try:
        outputs = model(**{k: v for k, v in batch.items()
                           if k in ("input_ids", "attention_mask", "labels")})
    except TypeError:  # pragma: no cover
        outputs = model(input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"))
    if getattr(outputs, "loss", None) is not None:
        return outputs.loss
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    labels = batch["labels"]
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1).to(shift_logits.device),
        ignore_index=-100,
    )


def _unlearning_loss(model: Any, batch: Any) -> Any:
    """SEAM unlearning loss ``L_ul`` (paper Eq. 3; official ``split="harmful"``).

    RepNoise-style layer-wise gradient ascent: read out masked next-token CE
    from *every* hidden state via the final norm + LM head, average over
    layers, and return ``-log(mean_ce + 1)`` so descending it raises harmful CE.

    Falls back to plain ``-log(CE + 1)`` on the final logits when hidden states
    or the norm / output-embeddings handles are unavailable.
    """
    import torch.nn.functional as F

    def masked_ce(logits, labels):
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1).to(shift_logits.device),
            ignore_index=-100,
        )

    labels = batch["labels"]
    try:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
        )
    except TypeError:  # pragma: no cover
        outputs = model(input_ids=batch["input_ids"], output_hidden_states=True)

    base_ce = masked_ce(outputs.logits, labels)

    hidden_states = getattr(outputs, "hidden_states", None)
    out_emb = None
    norm = None
    try:
        core = model.module if hasattr(model, "module") else model
        out_emb = core.get_output_embeddings()
        # Locate the final RMSNorm/LayerNorm (Qwen/Llama: model.model.norm).
        base = getattr(core, "base_model", None) or getattr(core, "model", None) or core
        norm = getattr(base, "norm", None)
        if norm is None and hasattr(base, "model"):
            norm = getattr(base.model, "norm", None)
    except Exception:  # pragma: no cover
        out_emb = None

    if hidden_states is not None and out_emb is not None and norm is not None:
        total = base_ce
        for h in hidden_states:
            projected = out_emb(norm(h).to(dtype=out_emb.weight.dtype))
            total = total + masked_ce(projected, labels)
        mean_ce = total / len(hidden_states) + 1.0
    else:
        logger.debug("SEAM L_ul: layer-wise read-out unavailable; using final-logit CE.")
        mean_ce = base_ce + 1.0

    return -torch.log(mean_ce)


def _flat_grad(model: Any) -> Dict[str, Any]:
    """Snapshot current ``.grad`` of every trainable param as a detached dict."""
    grad = {}
    params = (model.module.named_parameters()
              if hasattr(model, "module") else model.named_parameters())
    for name, p in params:
        if p.requires_grad:
            grad[name] = (p.grad.detach().clone() if p.grad is not None
                          else torch.zeros_like(p.data))
    return grad


def _grad_norm(grad: Dict[str, Any]) -> Any:
    return torch.sqrt(sum((g * g).sum() for g in grad.values()))


def _cosine(g1: Dict[str, Any], g2: Dict[str, Any]) -> Any:
    common = set(g1) & set(g2)
    dot = sum((g1[n] * g2[n]).sum() for n in common)
    n1 = torch.sqrt(sum((g1[n] * g1[n]).sum() for n in common))
    n2 = torch.sqrt(sum((g2[n] * g2[n]).sum() for n in common))
    return dot / (n1 * n2 + 1e-8)


# ---------------------------------------------------------------------------
# SEAMTrainer
# ---------------------------------------------------------------------------

class SEAMTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """HuggingFace Trainer implementing the faithful SEAM objective (Eq. 5).

    The total parameter gradient assembled each step is::

        grad = g_ul + alpha * g_up + (beta / eps) * grad_L_sd_hat

    where ``g_ul = grad L_ul`` (Eq. 3), ``g_up = grad L_up`` (Eq. 4), and
    ``grad_L_sd_hat`` is the Hessian-free finite-difference estimate of the
    gradient of the cosine-coupling term ``L_sd`` (Eqs. 2 & 6).

    Args:
        harmful_dataset:  ``D_adv`` — adversarial / harmful examples (required).
        alignment_dataset: ``D_aln`` — harmful-prompt / refusal pairs for the
            utility term ``L_up``.  Defaults to ``harmful_dataset`` when None.
        benign_dataset:   ``D_bgn`` — benign examples whose gradient is coupled
            against the harmful gradient.  Defaults to the ``train_dataset``
            batch when None.
    """

    def __init__(
        self,
        *args: Any,
        harmful_dataset: Optional[Any] = None,
        alignment_dataset: Optional[Any] = None,
        benign_dataset: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError("transformers is required for SEAMTrainer") from _TRAINER_IMPORT_ERROR
        if _TORCH_IMPORT_ERROR is not None:
            raise ImportError("torch is required for SEAMTrainer") from _TORCH_IMPORT_ERROR
        if harmful_dataset is None:
            raise ValueError("SEAMTrainer requires a 'harmful_dataset' argument (D_adv).")

        super().__init__(*args, **kwargs)

        self._alpha = float(getattr(self.args, "seam_alpha", 1.0))
        self._beta = float(getattr(self.args, "seam_beta", 0.001))
        self._epsilon = float(getattr(self.args, "seam_epsilon", 1e-3))
        self._refresh_every = max(1, int(getattr(self.args, "seam_refresh_every", 1)))

        # Deprecated alias: if seam_lambda was set to a non-default value, use
        # it as the coupling weight to preserve the old knob's intent.
        _lam = float(getattr(self.args, "seam_lambda", 0.5))
        if abs(_lam - 0.5) > 1e-12:
            logger.warning(
                "SEAMTrainer: 'seam_lambda' is deprecated; mapping it to "
                "'seam_beta' (gradient-coupling weight)."
            )
            self._beta = _lam

        try:
            _bs = self.args.per_device_train_batch_size
        except AttributeError:
            _bs = 1

        collate = getattr(self, "data_collator", None)
        self._harm_iter: Iterator = _make_cycling_loader(harmful_dataset, _bs, collate)
        self._align_iter: Optional[Iterator] = (
            _make_cycling_loader(alignment_dataset, _bs, collate)
            if alignment_dataset is not None else None
        )
        self._benign_iter: Optional[Iterator] = (
            _make_cycling_loader(benign_dataset, _bs, collate)
            if benign_dataset is not None else None
        )

        self._cached: Dict[str, Any] = {}
        self._step_count = 0
        # Last-step diagnostics (consumed by tests / logging).
        self.last_seam_metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def _next(self, it: Optional[Iterator], device: Any, fallback: Any) -> Any:
        if it is None:
            return fallback
        try:
            return _prepare_batch(next(it), device)
        except StopIteration:  # pragma: no cover
            return fallback

    def _refresh_batches(self, inputs: Any, device: Any) -> None:
        if self._cached and (self._step_count % self._refresh_every != 0):
            return
        benign = self._next(self._benign_iter, device, inputs)
        harmful = self._next(self._harm_iter, device, inputs)
        align = self._next(self._align_iter, device, harmful)
        self._cached = {"benign": benign, "harmful": harmful, "align": align}

    def _set_grads(self, model: Any, total_grad: Dict[str, Any]) -> None:
        """Accumulate the assembled SEAM gradient into ``param.grad``.

        Uses ``add_`` (not ``copy_``) so gradients from earlier micro-batches
        survive under gradient accumulation; the HF Trainer zeroes ``.grad``
        once per optimizer step, not per micro-batch.
        """
        params = (model.module.named_parameters()
                  if hasattr(model, "module") else model.named_parameters())
        for name, p in params:
            if not p.requires_grad:
                continue
            g = total_grad.get(name)
            if g is None:
                continue
            g = g.to(p.data.device)
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.add_(g)

    # ------------------------------------------------------------------
    # Faithful SEAM gradient assembly (paper Eq. 5 with Eq. 6 estimator).
    # ------------------------------------------------------------------
    def training_step(self, model, inputs, num_items_in_batch=None):  # type: ignore[override]
        model.train()
        device = next(model.parameters()).device
        inputs = _prepare_batch(inputs, device)
        self._refresh_batches(inputs, device)
        benign = self._cached["benign"]
        harmful = self._cached["harmful"]
        align = self._cached["align"]

        eps, alpha, beta = self._epsilon, self._alpha, self._beta
        total: Dict[str, Any] = {}

        def accumulate(grad: Dict[str, Any], scale: float) -> None:
            for n, g in grad.items():
                if n in total:
                    total[n] = total[n] + scale * g
                else:
                    total[n] = scale * g.clone()

        # Gradient-accumulation safety: the internal ``model.zero_grad()``
        # calls below isolate PROBE gradients (each backward's ``.grad`` is
        # read out immediately by ``_flat_grad``) and are required — but they
        # must not erase user gradients accumulated by earlier micro-batches.
        # Stash ``param.grad`` here and restore it before ``_set_grads``
        # accumulates the assembled SEAM gradient on top.
        _params = (model.module.parameters()
                   if hasattr(model, "module") else model.parameters())
        stashed_grads = [(p, p.grad) for p in _params if p.grad is not None]
        for p, _ in stashed_grads:
            p.grad = None

        try:
            # ---- g_ul = grad L_ul (Eq. 3, unlearning ascent) ----------------
            model.zero_grad()
            l_ul = _unlearning_loss(model, harmful)
            l_ul.backward()
            g_ul = _flat_grad(model)
            accumulate(g_ul, 1.0)

            # ---- g_up = grad L_up (Eq. 4, utility preservation) -------------
            model.zero_grad()
            l_up = _ce_loss(model, align)
            l_up.backward()
            g_up = _flat_grad(model)
            accumulate(g_up, alpha)

            # ---- L_sd coupling: gradients g_a (harmful-attack) & g_b (benign)
            model.zero_grad()
            l_a = _ce_loss(model, harmful)        # adversarial fine-tuning loss
            l_a.backward()
            g_a = _flat_grad(model)

            model.zero_grad()
            l_b = _ce_loss(model, benign)         # benign SFT loss
            l_b.backward()
            g_b = _flat_grad(model)

            na = _grad_norm(g_a)
            nb = _grad_norm(g_b)
            cos = _cosine(g_a, g_b)               # L_sd = sim(g_a, g_b) (Eq. 2)

            # ---- Hessian-free estimate of grad L_sd (Eq. 6) -----------------
            # Perturb 1: theta + eps*(gbar_a - c*gbar_b), then recompute g_b.
            g_b_pert = self._grad_at_perturbed(
                model, benign, g_a, g_b, na, nb, cos, eps, _ce_loss)
            # Perturb 2: theta + eps*(gbar_b - c*gbar_a), then recompute g_a.
            g_a_pert = self._grad_at_perturbed(
                model, harmful, g_b, g_a, nb, na, cos, eps, _ce_loss)

            sd_scale = beta / eps
            for n in total.keys() | g_a.keys():
                term = ((g_b_pert.get(n, 0.0) - g_b[n]) / (nb + 1e-12)
                        + (g_a_pert.get(n, 0.0) - g_a[n]) / (na + 1e-12))
                contrib = sd_scale * term
                total[n] = total[n] + contrib if n in total else contrib

            # ---- clear internal-pass residue, restore user grads, then ----
            # ---- accumulate the assembled gradient -------------------------
            model.zero_grad()
        finally:
            for p, g in stashed_grads:
                p.grad = g

        # Scale each micro-batch contribution by 1/gradient_accumulation_steps
        # so accumulated micro-steps average (mirroring the division applied
        # to the reported scalar loss below and HF's per-micro-batch scaling).
        ga_steps = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))
        if ga_steps > 1:
            total = {n: g / ga_steps for n, g in total.items()}
        self._set_grads(model, total)
        self._step_count += 1

        self.last_seam_metrics = {
            "L_ul": float(l_ul.detach().item()),
            "L_up": float(l_up.detach().item()),
            "L_sd_cos": float(cos.detach().item()),
            "g_a_norm": float(na.detach().item()),
            "g_b_norm": float(nb.detach().item()),
        }

        # Scalar reported to the Trainer loop (L_ul + alpha*L_up + beta*L_sd).
        reported = (l_ul.detach() + alpha * l_up.detach()
                    + beta * cos.detach())
        if self.args.n_gpu > 1:  # pragma: no cover
            reported = reported.mean()
        return reported / self.args.gradient_accumulation_steps

    def _grad_at_perturbed(self, model, batch, grad_dir, grad_sub,
                           norm_dir, norm_sub, cos, eps, loss_fn) -> Dict[str, Any]:
        """Compute grad of ``loss_fn`` at theta + eps*(gbar_dir - c*gbar_sub).

        Mirrors the official ``temporary_parameter_update`` context: perturb
        params along ``grad_dir/||grad_dir|| - c * grad_sub/||grad_sub||``,
        recompute the gradient, then restore the original params.
        """
        params = list(model.module.named_parameters()
                      if hasattr(model, "module") else model.named_parameters())
        original = {n: p.data.clone() for n, p in params if p.requires_grad}
        try:
            with torch.no_grad():
                for n, p in params:
                    if not p.requires_grad:
                        continue
                    direction = (grad_dir[n] / (norm_dir + 1e-12)
                                 - cos * grad_sub[n] / (norm_sub + 1e-12))
                    p.data.add_(direction.to(p.data.device), alpha=eps)
            model.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            return _flat_grad(model)
        finally:
            with torch.no_grad():
                for n, p in params:
                    if p.requires_grad:
                        p.data.copy_(original[n])
            model.zero_grad()
            del original

    # ------------------------------------------------------------------
    # compute_loss retained for API compatibility / inspection.
    # ------------------------------------------------------------------
    def compute_loss(  # type: ignore[override]
        self,
        model: Any,
        inputs: Any,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Any:
        """Scalar SEAM surrogate ``L_ul + alpha*L_up + beta*L_sd`` (Eq. 5).

        Note: the *true* SEAM parameter update is assembled in
        ``training_step`` using the Hessian-free estimator (Eq. 6); this scalar
        is differentiable in ``L_ul`` and ``L_up`` and uses the (non-Hessian)
        cosine value for the ``L_sd`` term, so it is suitable for inspection,
        logging and tests but is *not* the gradient applied during training.
        """
        device = next(model.parameters()).device
        inputs = _prepare_batch(inputs, device)
        self._refresh_batches(inputs, device)
        benign = self._cached["benign"]
        harmful = self._cached["harmful"]
        align = self._cached["align"]

        l_ul = _unlearning_loss(model, harmful)
        l_up = _ce_loss(model, align)

        g_a = self._grad_dict(model, harmful, _ce_loss)
        g_b = self._grad_dict(model, benign, _ce_loss)
        cos = _cosine(g_a, g_b)
        model.zero_grad()

        self.last_seam_metrics = {
            "L_ul": float(l_ul.detach().item()),
            "L_up": float(l_up.detach().item()),
            "L_sd_cos": float(cos.detach().item()),
        }

        loss = l_ul + self._alpha * l_up + self._beta * cos
        return (loss, None) if return_outputs else loss

    @staticmethod
    def _grad_dict(model, batch, loss_fn) -> Dict[str, Any]:
        model.zero_grad()
        loss = loss_fn(model, batch)
        loss.backward()
        return _flat_grad(model)


__all__ = ["SEAMConfig", "SEAMTrainer"]
