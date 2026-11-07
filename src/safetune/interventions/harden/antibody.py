"""
Antibody Trainer adapter.

Reference: "Antibody: Strengthening Defense Against Harmful Fine-Tuning for
Large Language Models via Attenuating Harmful Gradient Influence",
arXiv:2603.00498. Official code: https://github.com/minhquoc0712/Antibody (the paper
links it). This implementation follows the equations in the paper text; the official
alignment trainer uses a different, cosine-based adaptive weight (see ``xi`` below).

Antibody is a two-stage defense against harmful fine-tuning.

Alignment stage (``mode="align"``) -- SAM-style *flatness* regularization on
harmful samples. Along with the alignment loss the optimizer flattens the loss
landscape with respect to potential harmful samples so the instilled safety
behavior is hard to remove. The combined objective (paper Eq. 11) is::

    L_align(theta) + lambda_t * L_sharp(theta) + lambda_refusal * L_refusal(theta_pert)

where the sharpness loss is the harmful loss minus its minimum inside a
rho-ball (paper)::

    L_sharp(theta) = L_harm(theta) - min_{||phi-theta||<=rho} L_harm(phi)

The inner minimization is solved with a single normalized-gradient SAM step
(paper, K=1)::

    theta_pert = theta - rho * grad L_harm(theta) / ||grad L_harm(theta)||_2

so ``grad L_sharp(theta) ~= grad L_harm(theta) - grad L_harm(theta_pert)``.
The adaptive flatness weight follows the paper's Theorem 4.1::

    delta_t = grad L_align + lambda_t * grad L_sharp
    lambda_t = max{0, (a - <grad L_sharp, grad L_align>) / ||grad L_sharp||^2}

and a refusal loss ``L_refusal(theta_pert) = -sum log pi(y_r | x)`` is added on
the perturbed model so the refusal behavior survives the worst-case harmful
perturbation.

Fine-tuning stage (``mode="finetune"``) -- *sample-level* likelihood-ratio
reweighting. Each sample is scored by the log-likelihood ratio (paper Eq. 7)::

    r(x_i, y_i) = log pi_theta(y_i | x_i) - log pi_theta(y_r | x_i)

A high ``r`` means the model already prefers the user target over a refusal
(benign sample); a low/negative ``r`` means the model would rather refuse
(harmful sample). The per-batch softmax-normalized weight is::

    w_i = exp(r_i / tau) / sum_j exp(r_j / tau)

and the weighted update (paper Eq. 12) is::

    theta <- theta - eta * (1/L) * sum_i w_i * grad l(x_i, y_i)

so harmful samples' gradient contributions are down-weighted while benign ones
are up-weighted. The weights are detached (stop-gradient): they reweight the
per-sample losses but autograd does not differentiate through them.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

try:
    from transformers import Trainer, TrainingArguments
    _TRAINER_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    Trainer = object  # type: ignore[assignment,misc]
    TrainingArguments = object  # type: ignore[assignment,misc]
    _TRAINER_IMPORT_ERROR = _e

try:
    from safetune.core.optim.antibody import AntibodyConfig, AntibodyWrapper
    _ANTIBODY_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    AntibodyConfig = None  # type: ignore[assignment]
    AntibodyWrapper = None  # type: ignore[assignment]
    _ANTIBODY_IMPORT_ERROR = _e


class AntibodyTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """Trainer implementing Antibody's two-stage defense.

    Args:
        harmful_dataset: iterable of harmful batches used by the alignment-stage
            SAM flatness regularizer. Each item must be usable as
            ``model(**batch).loss``. Cycled indefinitely.
        refusal_dataset: optional iterable of refusal batches ``(x, y_r)`` used
            for the alignment-stage refusal loss on the perturbed model and as
            the default source of ``y_r`` for the fine-tuning-stage likelihood
            ratio. Cycled indefinitely.
        mode: ``"align"`` runs the SAM flatness alignment stage; ``"finetune"``
            runs the likelihood-ratio sample reweighting stage. Defaults to
            ``"align"``.
        refresh_every: how often (in steps) to draw a fresh harmful/refusal
            batch in the alignment stage.
        sam_rho: radius rho of the SAM perturbation ball. If ``None`` the value
            is read from ``args.sam_rho`` (``AntibodyConfig`` default 0.05).
        lambda_refusal: weight of the alignment-stage refusal loss.
        lambda_cap: upper clamp on the adaptive flatness weight ``lambda_t``.
        xi: safety margin for the adaptive flatness weight (paper Theorem 4.1:
            ``a_t = xi * ||g_sharp||²``). The default ``0.0`` recovers the
            simplified ``a_t=0`` formula; setting ``xi > 0`` (e.g. ``0.1``)
            guarantees the sharpness constraint is met even when gradients are
            orthogonal, matching the paper's dynamic threshold.
        tau: softmax temperature for the fine-tuning-stage reweighting.

    All extra arguments are optional keyword arguments with defaults, so the
    public ``AntibodyTrainer(...)`` signature is unchanged.
    """

    def __init__(
        self,
        *args: Any,
        harmful_dataset: Any = None,
        refusal_dataset: Any = None,
        mode: str = "align",
        refresh_every: int = 1,
        sam_rho: Optional[float] = None,
        lambda_refusal: float = 1.0,
        lambda_cap: float = 10.0,
        xi: float = 0.0,
        tau: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for AntibodyTrainer"
            ) from _TRAINER_IMPORT_ERROR
        if _ANTIBODY_IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.optim.antibody is unavailable"
            ) from _ANTIBODY_IMPORT_ERROR

        # The SAM steps call .backward() themselves and write the combined gradient into
        # .grad, so the fp16 GradScaler never scales the loss and the Trainer's later
        # unscale_ raises. Run these steps without it (fp32 master weights, no loss scaling).
        train_args = kwargs.get("args")
        if train_args is not None and getattr(train_args, "fp16", False):
            import dataclasses
            logger.warning("AntibodyTrainer does not support fp16 loss scaling; training with fp16=False.")
            kwargs["args"] = dataclasses.replace(train_args, fp16=False)

        super().__init__(*args, **kwargs)

        self._wrapper = AntibodyWrapper(self.model)
        self._mode = str(mode).lower()
        self._harmful_dataset = harmful_dataset
        self._refusal_dataset = refusal_dataset
        self._refresh_every = max(1, int(refresh_every))
        self._step_counter = 0

        rho = sam_rho if sam_rho is not None else getattr(self.args, "sam_rho", 0.05)
        self._sam_rho = float(rho)
        self._lambda_refusal = float(lambda_refusal)
        self._lambda_cap = float(lambda_cap)
        self._xi = float(xi)
        self._tau = max(1e-6, float(tau))

        self._harmful_iter: Optional[Iterator] = self._make_iter(self._harmful_dataset)
        self._refusal_iter: Optional[Iterator] = self._make_iter(self._refusal_dataset)

        # Cache of the most recently used harmful batch (alignment stage).
        self._last_harmful: Any = None
        # Whether the plain-SFT fallback warning has been emitted (H1 fix).
        self._fallback_warned = False

    # ------------------------------------------------------------------ utils
    def _make_iter(self, dataset: Any) -> Optional[Iterator]:
        if dataset is None:
            return None
        try:
            import torch

            bs = getattr(self.args, "per_device_train_batch_size", 1)
            loader = torch.utils.data.DataLoader(dataset, batch_size=bs)
            return iter(loader)
        except Exception:
            try:
                return iter(dataset)
            except Exception:
                return None

    def _next(self, which: str) -> Any:
        if which == "harmful":
            it, ds = self._harmful_iter, self._harmful_dataset
        else:
            it, ds = self._refusal_iter, self._refusal_dataset
        if it is None:
            return None
        try:
            batch = next(it)
        except StopIteration:
            it = self._make_iter(ds)
            if it is None:
                return None
            batch = next(it)
        if which == "harmful":
            self._harmful_iter = it
        else:
            self._refusal_iter = it
        return batch

    def _prepare(self, batch: Any, model) -> Any:
        if batch is None:
            return None
        try:
            if hasattr(self, "_prepare_inputs"):
                batch = self._prepare_inputs(batch)
        except Exception:
            pass
        if isinstance(batch, dict):
            batch = {
                k: (v.to(model.device) if hasattr(v, "to") else v)
                for k, v in batch.items()
            }
        return batch

    @staticmethod
    def _batch_loss(model, batch) -> Any:
        if isinstance(batch, dict):
            out = model(**batch)
        else:
            out = model(batch)
        return out.loss if hasattr(out, "loss") else out

    @staticmethod
    def _grad_dict(model):
        return {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        }

    @staticmethod
    def _global_norm(grads) -> Any:
        import torch

        if not grads:
            return torch.tensor(0.0)
        norms = torch.stack([g.norm(p=2) for g in grads.values()])
        return torch.norm(norms, p=2)

    # --------------------------------------------------- alignment stage (SAM)
    def _alignment_step(self, model, inputs, num_items_in_batch):
        """SAM flatness alignment step (paper Eqs. 10-11, Theorem 4.1)."""
        import torch

        harmful = self._prepare(self._next("harmful"), model)
        if harmful is None:
            # No harmful data -> fall back to a plain alignment step.
            return self._plain_step(model, inputs, num_items_in_batch)
        self._last_harmful = harmful

        # Gradient accumulation (H2 fix): the internal zero_grad calls below
        # would destroy gradients accumulated by earlier micro-steps, and the
        # final assignment used to overwrite instead of accumulate. Snapshot
        # the accumulated grads here and add them back in step 8.
        prior_grads = self._grad_dict(model)

        # 1) Alignment / SFT gradient g_align at theta.
        model.zero_grad(set_to_none=True)
        align_loss = self.compute_loss(model, inputs)
        align_loss.backward()
        g_align = self._grad_dict(model)

        # 2) Harmful gradient g_harm at theta.
        model.zero_grad(set_to_none=True)
        harm_loss = self._batch_loss(model, harmful)
        harm_loss.backward()
        g_harm = self._grad_dict(model)

        # 3) SAM perturbation: theta_pert = theta - rho * g_harm / ||g_harm||.
        harm_norm = self._global_norm(g_harm).clamp_min(1e-12)
        deltas = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                gh = g_harm.get(name)
                if gh is None:
                    continue
                delta = self._sam_rho * gh.to(p.dtype).to(p.device) / harm_norm.to(p.device)
                p.data.sub_(delta)
                deltas[name] = delta

        # 4) Harmful gradient at theta_pert -> grad L_harm(theta_pert).
        model.zero_grad(set_to_none=True)
        harm_loss_pert = self._batch_loss(model, harmful)
        harm_loss_pert.backward()
        g_harm_pert = self._grad_dict(model)

        # 4b) Refusal loss on the perturbed model: -sum log pi(y_r | x).
        g_refusal = {}
        refusal = self._prepare(self._next("refusal"), model)
        if refusal is not None and self._lambda_refusal != 0.0:
            model.zero_grad(set_to_none=True)
            try:
                refusal_loss = self._batch_loss(model, refusal)
                refusal_loss.backward()
                g_refusal = self._grad_dict(model)
            except Exception:
                g_refusal = {}

        # 5) Restore theta.
        with torch.no_grad():
            for name, p in model.named_parameters():
                d = deltas.get(name)
                if d is not None:
                    p.data.add_(d)

        # 6) grad L_sharp = grad L_harm(theta) - grad L_harm(theta_pert).
        g_sharp = {}
        for name, gh in g_harm.items():
            ghp = g_harm_pert.get(name)
            g_sharp[name] = gh - ghp if ghp is not None else torch.zeros_like(gh)

        # 7) Adaptive flatness weight (Theorem 4.1):
        #    a_t = xi * ||g_sharp||^2
        #    lambda_t = max{0, (a_t - <g_sharp, g_align>) / ||g_sharp||^2}
        #             = max{0, xi - <g_sharp, g_align> / ||g_sharp||^2}
        # xi=0 (default) recovers the simplified a_t=0 variant; xi>0 matches paper.
        dot = torch.tensor(0.0)
        sq = torch.tensor(0.0)
        for name, gs in g_sharp.items():
            ga = g_align.get(name)
            if ga is not None:
                dot = dot + torch.sum(gs.float() * ga.float().to(gs.device))
            sq = sq + torch.sum(gs.float() * gs.float())
        sq = sq.clamp_min(1e-12)
        lambda_t = torch.clamp(self._xi - dot / sq, min=0.0, max=self._lambda_cap)
        self._last_lambda = float(lambda_t)

        # 8) Final gradient: g_align + lambda_t * g_sharp + lambda_r * g_refusal.
        # Scale by the number of gradient-accumulation micro-steps (mirroring
        # HF Trainer's own loss scaling) and *accumulate* onto any gradients
        # from earlier micro-steps instead of overwriting them (H2 fix).
        # With gradient_accumulation_steps == 1 this is identical to before.
        ga_steps = max(
            1,
            int(
                getattr(
                    self,
                    "current_gradient_accumulation_steps",
                    getattr(self.args, "gradient_accumulation_steps", 1),
                )
            ),
        )
        with torch.no_grad():
            for name, p in model.named_parameters():
                ga = g_align.get(name)
                prior = prior_grads.get(name)
                if ga is None:
                    # No new gradient for this param; restore any accumulated
                    # gradient wiped by the internal zero_grad calls.
                    if prior is not None and p.grad is None:
                        p.grad = prior
                    continue
                final = ga.float()
                gs = g_sharp.get(name)
                if gs is not None:
                    final = final + lambda_t.to(gs.device) * gs.float()
                gr = g_refusal.get(name)
                if gr is not None:
                    final = final + self._lambda_refusal * gr.float()
                if ga_steps > 1:
                    final = final / ga_steps
                if prior is not None:
                    final = prior.float() + final
                p.grad = final.to(p.dtype)

        return align_loss.detach()

    # ------------------------------------- fine-tuning stage (LR reweighting)
    def _finetune_step(self, model, inputs):
        """Likelihood-ratio sample reweighting step (paper Eqs. 7, 12)."""
        import torch

        if not isinstance(inputs, dict) or "labels" not in inputs:
            # Cannot compute per-sample ratios -> plain step.
            return self._plain_step(model, inputs, None)

        labels = inputs["labels"]
        if labels.dim() < 2 or labels.size(0) < 2:
            return self._plain_step(model, inputs, None)

        # NOTE: no model.zero_grad() here (H2 fix) -- HF Trainer zeroes
        # gradients at optimizer-step boundaries; zeroing per micro-step
        # destroyed gradient accumulation.

        # Per-sample log pi(y_i | x_i) on the user target.
        out = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = out.logits  # (B, T, V)
        target_logp = self._sequence_logprob(logits, labels)  # (B,)

        # Per-sample log pi(y_r | x_i) on a refusal target.
        refusal_labels = self._refusal_labels(inputs, model)
        if refusal_labels is not None:
            with torch.no_grad():
                ref_logp = self._refusal_logprob(model, inputs, logits, refusal_labels)
        else:
            # No refusal reference -> uniform weights (degrades to plain SFT).
            ref_logp = torch.zeros_like(target_logp)

        # Likelihood ratio r_i and softmax-normalized weights (detached).
        with torch.no_grad():
            r = target_logp.detach() - ref_logp
            w = torch.softmax(r / self._tau, dim=0)  # (B,), sums to 1
        self._last_weights = w.detach().cpu()

        # Per-sample CE losses l(x_i, y_i).
        per_sample_loss = self._per_sample_ce(logits, labels)  # (B,)

        # Weighted update: (1/L) sum_i w_i * l_i. Scale by B so the magnitude
        # matches a normal mean-reduced batch loss (sum w_i = 1).
        weighted_loss = torch.sum(w * per_sample_loss)

        # Single gradient-accumulation scaling (H2): this override bypasses
        # HF Trainer.training_step (which normally does this division), and
        # the Trainer configures its Accelerator with num_steps=1, so
        # accelerator.backward does NOT divide -- this is the only division.
        if self.args.gradient_accumulation_steps > 1:
            weighted_loss = weighted_loss / self.args.gradient_accumulation_steps

        if getattr(self, "accelerator", None) is not None:
            self.accelerator.backward(weighted_loss)
        else:
            weighted_loss.backward()

        return weighted_loss.detach()

    def _refusal_logprob(self, model, inputs, logits, refusal_labels):
        """Per-sample ``log pi(y_r | x_i)`` for the likelihood ratio (paper Eq. 7).

        ``y_r`` must be scored after the sample's *own* prompt ``x_i``. The official
        data pipeline pairs every instance with a refusal view built from the same
        source prompt (minhquoc0712/Antibody @6c557b1, ``utils.py:347-360`` cycles the
        refusal answers over the instances and ``utils.py:388-394`` tokenises
        ``preprocess(sources, [refusal + eos])``). Here the refusal batch only supplies
        the *answer* tokens (positions where ``refusal_labels != -100``, row ``i`` uses
        refusal row ``i mod B_r``); each sample's prompt (its tokens before the first
        supervised label) is re-used, and ``(x_i, y_r)`` is scored in a second no-grad
        forward. Falls back to the same-forward gather only when the batch has no
        ``input_ids`` (e.g. ``inputs_embeds``-only).
        """
        import torch

        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
        labels = inputs.get("labels") if isinstance(inputs, dict) else None
        if (
            input_ids is None
            or labels is None
            or not hasattr(input_ids, "shape")
            or input_ids.dim() != 2
            or refusal_labels.shape[:1] == (0,)
        ):
            # Cannot rebuild (x_i, y_r); degrade to the old (miscalibrated) gather.
            return self._sequence_logprob(logits.detach(), refusal_labels)

        device = input_ids.device
        attn = inputs.get("attention_mask")
        if attn is None:
            attn = torch.ones_like(input_ids)
        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        pad_id = getattr(tok, "pad_token_id", None)
        pad_id = 0 if pad_id is None else int(pad_id)

        rows_ids, rows_mask, rows_lab = [], [], []
        n_ref = refusal_labels.size(0)
        for i in range(input_ids.size(0)):
            supervised = (labels[i] != -100).nonzero(as_tuple=False)
            start = int(supervised[0]) if len(supervised) else int(input_ids.size(1))
            r = refusal_labels[i % n_ref].to(device)
            answer = r[r != -100].to(input_ids.dtype)
            rows_ids.append(torch.cat([input_ids[i, :start], answer]))
            rows_mask.append(torch.cat([attn[i, :start], torch.ones_like(answer)]).to(attn.dtype))
            rows_lab.append(torch.cat([
                torch.full((start,), -100, dtype=torch.long, device=device),
                answer.to(torch.long),
            ]))
        width = max(int(x.numel()) for x in rows_ids)

        def _pad(rows, value, dtype):
            out = torch.full((len(rows), width), value, dtype=dtype, device=device)
            for k, row in enumerate(rows):
                out[k, : row.numel()] = row
            return out

        ref_input_ids = _pad(rows_ids, pad_id, input_ids.dtype)
        ref_mask = _pad(rows_mask, 0, attn.dtype)
        ref_labels = _pad(rows_lab, -100, torch.long)
        ref_out = model(input_ids=ref_input_ids, attention_mask=ref_mask)
        return self._sequence_logprob(ref_out.logits, ref_labels)

    @staticmethod
    def _sequence_logprob(logits, labels):
        """Sum of token log-probabilities of ``labels`` under ``logits``.

        Returns a (B,) tensor; positions with label -100 are ignored.
        """
        import torch
        import torch.nn.functional as F

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        logp = F.log_softmax(shift_logits.float(), dim=-1)
        mask = shift_labels != -100
        gather_labels = shift_labels.clamp_min(0).unsqueeze(-1)
        token_logp = torch.gather(logp, -1, gather_labels).squeeze(-1)
        token_logp = token_logp * mask
        return token_logp.sum(dim=-1)

    @staticmethod
    def _per_sample_ce(logits, labels):
        """Mean per-token cross-entropy for each sample; returns a (B,) tensor."""
        import torch.nn.functional as F

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        B, T, V = shift_logits.shape
        ce = F.cross_entropy(
            shift_logits.reshape(-1, V).float(),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(B, T)
        mask = (shift_labels != -100).float()
        denom = mask.sum(dim=-1).clamp_min(1.0)
        return (ce * mask).sum(dim=-1) / denom

    def _refusal_labels(self, inputs, model):
        """Resolve the refusal answers ``y_r`` used in the likelihood ratio.

        Priority: explicit ``refusal_labels`` field in the batch, else a batch drawn from
        ``refusal_dataset``. Only the answer tokens (labels != -100) are used, so the
        batch does not need to match the user batch in shape or prompt. Returns ``None``
        if no refusal reference is available.
        """
        if "refusal_labels" in inputs:
            return inputs["refusal_labels"].to(inputs["labels"].device)

        batch = self._prepare(self._next("refusal"), model)
        if isinstance(batch, dict) and "labels" in batch:
            return batch["labels"].to(inputs["labels"].device)
        return None

    # ----------------------------------------------------------------- common
    def _plain_step(self, model, inputs, num_items_in_batch):
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except TypeError:
            return super().training_step(model, inputs)

    def training_step(self, model, inputs, num_items_in_batch=None):  # type: ignore[override]
        model.train()
        # Standard HF Trainer pattern: move the main batch to the right device
        # before any custom forward (H1 fix -- previously the raw CPU batch hit
        # the GPU model, and the except below silently degraded every GPU run
        # to plain SFT).
        inputs = self._prepare_inputs(inputs)
        try:
            if self._mode == "finetune":
                loss = self._finetune_step(model, inputs)
            else:
                loss = self._alignment_step(model, inputs, num_items_in_batch)
        except Exception as e:
            # Any failure in the defense path falls back to a plain step so
            # training does not abort -- but loudly, so a systematic failure
            # (which means *zero* Antibody defense) cannot go unnoticed.
            if not self._fallback_warned:
                logger.warning(
                    "Antibody step failed (%s); falling back to plain SFT step",
                    e,
                )
                self._fallback_warned = True
            else:
                logger.debug(
                    "Antibody step failed (%s); falling back to plain SFT step",
                    e,
                )
            model.zero_grad(set_to_none=True)
            loss = self._plain_step(model, inputs, num_items_in_batch)
        self._step_counter += 1
        return loss


__all__ = ["AntibodyTrainer", "AntibodyConfig"]
