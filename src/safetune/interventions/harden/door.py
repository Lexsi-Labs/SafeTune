"""
DOOR Trainer adapter — faithful standalone implementation.

Faithful implementation of the DOOR / W-DOOR objective from:

    "Improving LLM Safety Alignment with Dual-Objective Optimization"
    Zhao, Cai, Shi, Huang, Lin, Mei, Song. ICML 2025. arXiv:2503.03710.
    Reference code: https://github.com/wicai24/DOOR-Alignment
      (``utils/loss.py`` -> ``gd_npo_loss`` for DOOR, ``wgdnpo_loss`` for W-DOOR)

The reference implementation (``utils/loss.py`` + a plain ``Trainer`` subclass)
uses **DOOR alone** as the training objective — not DPO + DOOR. SafeTune
matches this by default: when ``door_pure_mode=True`` (the default),
``compute_loss`` returns **only** the DOOR term and never calls the parent DPO
loss. Set ``door_pure_mode=False`` to revert to the legacy hybrid that mixes
DPO + DOOR (which deviates from the paper).

DOOR disentangles the DPO objective into two components:

  (1) Robust refusal -- MLE of the safe / refusal response,
      ``safety_loss = -log pi(y_safe | x)``.
  (2) Targeted unlearning of harmful knowledge -- the **NPO** loss, *not*
      naive negative cross-entropy (gradient ascent). The authors' term is

          npo = -(2/beta) * logsigmoid(-beta * (logp_pi(y_h) - logp_ref(y_h)))

      i.e. a sigmoid of the policy-vs-reference log-ratio on the harmful
      response. This is bounded and reference-anchored, avoiding the
      destabilisation of plain gradient ascent.

W-DOOR (``wgdnpo_loss``) additionally reweights the *refusal* tokens with a
weight derived from a **proxy reward model** -- a DPO-aligned policy whose
log-probabilities, relative to the reference, define a per-token reward:

      reward_t = logp_dpo(y_safe_t) - logp_ref(y_safe_t)
      sigmoid : w_t = 1 - sigmoid(gamma * reward_t)
      exp     : w_t = exp(-reward_t / tau)

The reward model is *learned* (a separately DPO-trained policy), not a
keyword list.

This adapter (``SafetyDOORTrainer``, re-exported as ``DOORTrainer`` in
:mod:`safetune.harden`) is a thin :class:`trl.DPOTrainer` subclass. It runs
the policy (and the frozen reference / optional proxy reward model) on the
DOOR batch fields itself, so the DOOR term fires on every batch that carries
those fields -- it no longer depends on logits being pre-baked into the batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Optional

try:
    from trl import DPOTrainer, DPOConfig
    _TRL_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    DPOTrainer = object  # type: ignore[assignment,misc]
    DPOConfig = object  # type: ignore[assignment,misc]
    _TRL_IMPORT_ERROR = _e


if _TRL_IMPORT_ERROR is None:
    @dataclass
    class DOORConfig(DPOConfig):  # type: ignore[misc]
        """DPOConfig subclass exposing the DOOR / W-DOOR hyper-parameters.

        All fields have defaults so the public construction signature is
        unchanged.

        ``door_pure_mode=True`` (default) is the **faithful** setting: only the
        DOOR term is used as the loss, matching the paper's ``gd_npo_loss``.
        Set ``door_pure_mode=False`` to mix DPO + DOOR (legacy hybrid, deviates
        from paper).
        """

        # When True (default, faithful): return pure DOOR loss; skip DPO parent.
        door_pure_mode: bool = True
        # Temperature of the NPO sigmoid (paper default beta = 0.5).
        door_beta: float = 0.5
        # Relative weight of the unlearning (NPO) term inside the DOOR term.
        unlearning_weight: float = 1.0
        # Relative weight of the robust-refusal term inside the DOOR term.
        refusal_weight: float = 1.0
        # Legacy hybrid mixing weight (only used when door_pure_mode=False).
        door_alpha: float = 0.5
        # W-DOOR token reweighting: enabled only when a proxy reward model is
        # supplied. "sigmoid" -> w = 1 - sigmoid(gamma * reward); "exp" ->
        # w = exp(-reward / tau).
        wdoor_method: str = "sigmoid"
        wdoor_gamma: float = 1.0
        wdoor_tau: float = 5.0
else:  # pragma: no cover
    class DOORConfig(object):  # type: ignore[assignment]
        pass


def _get_sequence_log_probs(logits: Any, labels: Any) -> Any:
    """Sum of per-token log-probabilities of ``labels`` under ``logits``.

    Mirrors ``utils/loss.py::get_sequence_log_probs`` in the authors' repo:
    next-token shift, ``ignore_index=-100``, summed over the sequence.
    """
    shifted_labels = labels[..., 1:].contiguous()
    shifted_logits = logits[..., :-1, :].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    # transpose -> (batch, vocab, seq) as CrossEntropyLoss expects.
    token_log_probs = -loss_fn(shifted_logits.transpose(-1, -2), shifted_labels)
    return token_log_probs.sum(dim=-1)


def _get_token_log_probs(logits: Any, labels: Any) -> Any:
    """Per-token log-probabilities (un-summed).

    Mirrors ``utils/loss.py::get_log_probs`` (= ``-get_batch_loss``); used
    for W-DOOR token-level reward weighting.
    """
    shifted_labels = labels[..., 1:].contiguous()
    shifted_logits = logits[..., :-1, :].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    return -loss_fn(shifted_logits.transpose(-1, -2), shifted_labels)


class SafetyDOORTrainer(DPOTrainer if _TRL_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """:class:`trl.DPOTrainer` with the DOOR / W-DOOR safety term mixed in.

    Optional ``proxy_reward_model`` enables W-DOOR: a DPO-aligned policy used
    to derive per-token refusal weights (the paper's reward model). When it
    is not supplied the trainer runs plain DOOR (un-weighted refusal MLE).
    """

    def __init__(
        self,
        *args: Any,
        proxy_reward_model: Any = None,
        **kwargs: Any,
    ) -> None:
        if _TRL_IMPORT_ERROR is not None:
            raise ImportError(
                "trl is required for SafetyDOORTrainer"
            ) from _TRL_IMPORT_ERROR

        super().__init__(*args, **kwargs)

        self._door_pure_mode = bool(getattr(self.args, "door_pure_mode", True))
        self._door_alpha = float(getattr(self.args, "door_alpha", 0.5))
        self._door_beta = float(getattr(self.args, "door_beta", 0.5))
        self._door_unlearn_w = float(getattr(self.args, "unlearning_weight", 1.0))
        self._door_refusal_w = float(getattr(self.args, "refusal_weight", 1.0))
        self._wdoor_method = str(getattr(self.args, "wdoor_method", "sigmoid"))
        self._wdoor_gamma = float(getattr(self.args, "wdoor_gamma", 1.0))
        self._wdoor_tau = float(getattr(self.args, "wdoor_tau", 5.0))

        # Proxy reward model for W-DOOR (a separately DPO-aligned policy).
        # Frozen and used only to score refusal tokens.
        self._proxy_reward_model = proxy_reward_model
        if self._proxy_reward_model is not None:
            self._proxy_reward_model.eval()
            for p in self._proxy_reward_model.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # DOOR / W-DOOR loss
    # ------------------------------------------------------------------
    def _npo_unlearning_loss(self, model: Any, inputs: dict) -> Any:
        """NPO unlearning term on the harmful response.

        Faithful to ``gd_npo_loss`` in ``utils/loss.py``:

            npo = -(2/beta) * logsigmoid(-beta * (logp_pi - logp_ref))

        evaluated on the harmful (non-preferred) response, averaged over the
        batch. This is *not* negative cross-entropy.
        """
        harmful_ids = inputs["harmful_input_ids"]
        harmful_labels = inputs["harmful_labels"]
        harmful_mask = inputs.get("harmful_attention_mask")

        policy_out = model(harmful_ids, attention_mask=harmful_mask)
        logp_main = _get_sequence_log_probs(policy_out.logits, harmful_labels)

        ref_model = getattr(self, "ref_model", None)
        
        if ref_model is not None:
            with torch.no_grad():
                ref_out = ref_model(harmful_ids, attention_mask=harmful_mask)
                logp_ref = _get_sequence_log_probs(ref_out.logits, harmful_labels)
        elif hasattr(model, "disable_adapter"):
            # LoRA / PEFT support: extract reference logits by disabling the adapter
            with torch.no_grad(), model.disable_adapter():
                ref_out = model(harmful_ids, attention_mask=harmful_mask)
                logp_ref = _get_sequence_log_probs(ref_out.logits, harmful_labels)
        else:
            # No reference model available -> reference-free fallback
            # (NPO reduces to a sigmoid of the policy log-prob alone).
            logp_ref = torch.zeros_like(logp_main)

        beta = self._door_beta
        npo = -F.logsigmoid(-beta * (logp_main - logp_ref)) * (2.0 / beta)
        return npo.mean()

    def _refusal_loss(self, model: Any, inputs: dict) -> Any:
        """Robust-refusal term (MLE of the safe response).

        Plain DOOR: ``safety_loss = -mean(sum_t logp(y_safe_t))``.
        W-DOOR: token-level reweighting by a proxy-reward-model weight,
        ``gd_loss = -mean(sum_t w_t * logp(y_safe_t))`` -- faithful to
        ``wgdnpo_loss`` (the reward is ``logp_dpo - logp_ref`` per token).
        """
        refusal_ids = inputs["refusal_input_ids"]
        refusal_labels = inputs["refusal_labels"]
        refusal_mask = inputs.get("refusal_attention_mask")

        policy_out = model(refusal_ids, attention_mask=refusal_mask)

        # ---- Plain DOOR: un-weighted refusal MLE -------------------------
        if self._proxy_reward_model is None:
            logp = _get_sequence_log_probs(policy_out.logits, refusal_labels)
            return -logp.mean()

        # ---- W-DOOR: proxy-reward-model token weighting ------------------
        token_logp_main = _get_token_log_probs(policy_out.logits, refusal_labels)

        ref_model = getattr(self, "ref_model", None)
        with torch.no_grad():
            proxy_out = self._proxy_reward_model(refusal_ids, attention_mask=refusal_mask)
            token_logp_dpo = _get_token_log_probs(proxy_out.logits, refusal_labels)
            
            if ref_model is not None:
                ref_out = ref_model(refusal_ids, attention_mask=refusal_mask)
                token_logp_ref = _get_token_log_probs(ref_out.logits, refusal_labels)
            elif hasattr(model, "disable_adapter"):
                # LoRA / PEFT support
                with model.disable_adapter():
                    ref_out = model(refusal_ids, attention_mask=refusal_mask)
                    token_logp_ref = _get_token_log_probs(ref_out.logits, refusal_labels)
            else:
                token_logp_ref = torch.zeros_like(token_logp_dpo)

            # Per-token reward from the proxy reward model.
            reward = token_logp_dpo - token_logp_ref
            
            if self._wdoor_method == "exp":
                # exp_weights: w = exp((ref - dpo) / tau) = exp(-reward/tau)
                weights = torch.exp(-reward / self._wdoor_tau)
            else:  # "sigmoid"
                # sigmoid_weights: w = 1 - sigmoid(gamma * reward)
                weights = 1.0 - torch.sigmoid(self._wdoor_gamma * reward)

        weighted_logp = (weights * token_logp_main).sum(dim=-1)
        return -weighted_logp.mean()

    def _compute_door_term(self, model: Any, inputs: dict) -> Any:
        """Combined DOOR term: refusal MLE + NPO unlearning."""
        refusal = self._refusal_loss(model, inputs)
        unlearn = self._npo_unlearning_loss(model, inputs)
        return self._door_refusal_w * refusal + self._door_unlearn_w * unlearn

    @staticmethod
    def _has_door_fields(inputs: Any) -> bool:
        """True if the batch carries the raw DOOR refusal+harmful tensors."""
        if not isinstance(inputs, dict):
            return False
        required = (
            "refusal_input_ids",
            "refusal_labels",
            "harmful_input_ids",
            "harmful_labels",
        )
        return all(k in inputs for k in required)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        if self._door_pure_mode:
            # Faithful path: pure DOOR objective, matching the paper's
            # ``gd_npo_loss`` (refusal MLE + NPO unlearning). The DPO parent
            # loss is intentionally not called here.
            if not self._has_door_fields(inputs):
                # Fallback: no DOOR fields in batch — return a zero loss.
                dummy = next(iter(model.parameters()))
                loss = dummy.sum() * 0.0
                return (loss, None) if return_outputs else loss
            door_term = self._compute_door_term(model, inputs)
            return (door_term, None) if return_outputs else door_term

        # Legacy hybrid path (door_pure_mode=False): DPO + DOOR mix.
        # This deviates from the paper — use only for backward compatibility.
        try:
            result = super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        except TypeError:
            result = super().compute_loss(model, inputs, return_outputs=return_outputs)

        if not self._has_door_fields(inputs):
            return result

        door_term = self._compute_door_term(model, inputs)

        if return_outputs:
            loss, outputs = result
            return (loss + self._door_alpha * door_term, outputs)
        return result + self._door_alpha * door_term