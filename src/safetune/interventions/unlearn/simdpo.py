"""
DPO Unlearning: Reference-Free (SimDPO) variant for safety unlearning.

Background
----------
Standard DPO unlearning (multiple 2024 papers) frames unlearning as a
preference optimisation problem: the harmful response is the ``rejected``
completion and a safe refusal is the ``chosen`` completion.  The DPO loss
then pushes the model *away* from the harmful output and *toward* the safe
refusal, simultaneously.

The reference-model variant requires loading a frozen copy of the model to
compute a baseline log-likelihood ratio, which doubles GPU memory.

``SimDPO`` here is SafeTune's name for **reference-free DPO unlearning using
the length-normalized implicit reward of SimPO** (Meng, Xia, Chen, "SimPO:
Simple Preference Optimization with a Reference-Free Reward", arXiv:2405.14734,
NeurIPS 2024). It removes the reference model by using the model's own
sequence-mean log-likelihood as the implicit reward:

  reward(y | x) = (β/|y|) * log p_θ(y | x)

  loss = -E[ logsigmoid( reward(y_c | x) - reward(y_r | x) - γ ) ]

       = -E[ logsigmoid( β/|y| * (log p_θ(y_c | x) - log p_θ(y_r | x)) - γ ) ]

FIDELITY: SimPO's defining **target reward margin γ** defaults to ``0`` here
(plain reference-free contrastive unlearning with SimPO's length-normalized
reward); set ``gamma`` > 0 to recover the full SimPO objective. This is also
distinct from SimNPO (a forget-only, no-pair length-normalized NPO variant).

For **safety unlearning** we adapt the polarity:

  * ``chosen``  = safe refusal response  (we want the model to output this).
  * ``rejected`` = harmful response      (we want the model to move away from
                                          this).

The loss then drives the harmful response's reward below the safe refusal's
reward, without any reference model.

Because the same forward pass through the model produces logits for both the
chosen and rejected sequences (they share the same prompt but differ in the
response portion), each batch supplies a *pair* ``(chosen_batch,
rejected_batch)`` of tokenized examples that cover ``[prompt + chosen_resp]``
and ``[prompt + rejected_resp]`` respectively.

Retain term
~~~~~~~~~~~
An optional retain term adds standard CE on benign data to prevent capability
collapse.  Enabled by default (``variant="simdpo_retain"``).  The bare
``"simdpo"`` variant omits it.

Data format
~~~~~~~~~~~
Forget batches must be pairs: each element is a ``dict`` with keys
``"chosen"`` and ``"rejected"``, each of which is itself a
``dict[str, Tensor]`` with ``input_ids``, ``attention_mask``, ``labels``
(``-100`` at positions not part of the response).

Retain batches (for ``simdpo_retain``) are plain single-example batches with
``input_ids``, ``attention_mask``, ``labels``.

References
----------
- Chen et al., "SimDPO: Simple Preference Optimization with a Reference-Free
  Reward", arXiv:2405.14734, 2024.
- Multiple DPO-for-unlearning papers, 2024:
  - Lu et al., "Online DPO for LLM Unlearning", 2024.
  - Zhang et al., "Negative Preference Optimization", arXiv:2404.05868, 2024.
  - Mazeika et al., "HarmBench", 2024 (DPO-based unlearning ablation).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_VARIANTS = ("simdpo", "simdpo_retain")


@dataclass
class SimDPOUnlearnConfig:
    """Configuration for SimDPO-based safety unlearning.

    Attributes:
        variant: ``"simdpo"`` — forget (pair) loss only;
            ``"simdpo_retain"`` — forget loss + retain CE.
        beta: SimDPO temperature.  ``0.1`` is a conservative default;
            larger values make the loss more sensitive to the reward gap.
        retain_coeff: weight on the retain CE term.  ``1.0`` is the
            standard equal-weight default.
        epochs: passes over the forget (and retain) iterable.
        lr: AdamW learning rate.  ``1e-5`` matches TOFU / NPO defaults.
        weight_decay: AdamW weight decay.  ``0.01`` matches TOFU defaults.
        max_steps: optional hard cap on total optimizer steps.  ``None``
            runs the full epoch schedule.
    """

    variant: str = "simdpo_retain"
    beta: float = 0.1
    retain_coeff: float = 1.0
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 0.01
    max_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.variant not in _VARIANTS:
            raise ValueError(
                f"SimDPOUnlearnConfig.variant must be one of {_VARIANTS}, "
                f"got {self.variant!r}"
            )


def _sequence_logp(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Per-sequence mean log-probability over non-masked response tokens.

    SimDPO uses the length-normalised log-likelihood as the implicit reward:
    ``reward = (1/|y|) * log p(y|x)``.  We compute ``log p`` as the mean
    per-token NLL (negated), where ``|y|`` is the number of non-masked
    response tokens.

    Args:
        logits: ``(batch, seq_len, vocab)`` model output logits.
        labels: ``(batch, seq_len)`` with ``-100`` at masked positions.

    Returns:
        ``(batch,)`` tensor of per-sequence mean log-probabilities.
        Sequences with no valid label tokens return ``0.0``.
    """
    shift_logits = logits[..., :-1, :].contiguous()   # (B, L-1, V)
    shift_labels = labels[..., 1:].contiguous()        # (B, L-1)

    mask = shift_labels != -100                        # (B, L-1)
    token_counts = mask.sum(dim=-1).float().clamp(min=1)  # (B,)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    per_token_nll = loss_fn(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    ).reshape(shift_labels.shape)                       # (B, L-1)

    # Sum NLL over response tokens, then normalise by response length.
    per_seq_nll = (per_token_nll * mask.float()).sum(dim=-1) / token_counts  # (B,)

    # log p = -NLL.
    return -per_seq_nll                                 # (B,)


def simdpo_forget_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    beta: float,
    gamma: float = 0.0,
) -> torch.Tensor:
    """Reference-free (SimPO-reward) contrastive loss for unlearning.

    Drives ``rejected_logp`` (harmful) below ``chosen_logp`` (safe refusal)
    using SimPO's length-normalized reference-free reward.

    ``loss = -E[ logsigmoid( beta * (chosen_logp - rejected_logp) - gamma ) ]``

    Minimising this loss increases the reward gap between the safe refusal
    and the harmful completion, without needing a frozen reference model.

    Args:
        chosen_logp: ``(batch,)`` per-sequence mean log-prob for the safe
            refusal (the ``chosen`` completion).
        rejected_logp: ``(batch,)`` per-sequence mean log-prob for the
            harmful response (the ``rejected`` completion).
        beta: temperature on the reward gap.
        gamma: SimPO target reward margin. ``0`` (default) gives plain
            reference-free DPO; ``> 0`` recovers the full SimPO objective.

    Returns:
        Scalar loss.
    """
    logits = beta * (chosen_logp - rejected_logp) - gamma
    return -F.logsigmoid(logits).mean()


def _build_optimizer(
    model: nn.Module, cfg: SimDPOUnlearnConfig
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def simdpo_unlearn(
    model: nn.Module,
    forget_batches: Iterable[Dict[str, Dict[str, torch.Tensor]]],
    retain_batches: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
    *,
    config: Optional[SimDPOUnlearnConfig] = None,
) -> nn.Module:
    """Run SimDPO-based safety unlearning in place on ``model``.

    No frozen reference model is needed — SimDPO uses the model's own
    length-normalised log-likelihood as the implicit reward baseline.

    Args:
        model: the model to unlearn.  Updated in place.
        forget_batches: iterable of *paired* forget batches.  Each element
            must be a ``dict`` with two keys:

            * ``"chosen"``   — batch for the safe refusal response.
            * ``"rejected"`` — batch for the harmful response.

            Both sub-batches must have ``input_ids``, ``attention_mask``,
            and ``labels`` (``-100`` at prompt / masked positions).

            Re-iterated once per epoch; pass a re-iterable for
            ``epochs > 1``.

        retain_batches: iterable of plain retain batches (required for
            ``simdpo_retain``; ignored for ``simdpo``).  Each batch has
            ``input_ids``, ``attention_mask``, ``labels``.  Re-iterated
            once per epoch and consumed one-per-forget-batch.

        config: :class:`SimDPOUnlearnConfig`.

    Returns:
        The updated model (same object, mutated in place).
    """
    cfg = config or SimDPOUnlearnConfig()
    needs_retain = cfg.variant == "simdpo_retain"
    if needs_retain and retain_batches is None:
        raise ValueError(
            "SimDPOUnlearnConfig.variant='simdpo_retain' requires retain_batches."
        )

    def _logits(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = model(**{k: v for k, v in batch.items() if k != "labels"})
        return out.logits if hasattr(out, "logits") else out

    opt = _build_optimizer(model, cfg)
    total_steps = 0
    max_steps = cfg.max_steps

    for epoch in range(1, max(1, cfg.epochs) + 1):
        if max_steps is not None and total_steps >= max_steps:
            break
        steps = 0
        retain_iter = iter(retain_batches) if needs_retain else None

        for f_pair in forget_batches:
            if max_steps is not None and total_steps >= max_steps:
                break

            # Validate pair format.
            if "chosen" not in f_pair or "rejected" not in f_pair:
                raise ValueError(
                    "simdpo_unlearn: each forget batch must be a dict with "
                    "'chosen' and 'rejected' keys.  Got: "
                    f"{list(f_pair.keys())}"
                )

            chosen_batch = f_pair["chosen"]
            rejected_batch = f_pair["rejected"]

            opt.zero_grad(set_to_none=True)

            # --- SimDPO forget loss -----------------------------------------
            # Forward pass for chosen (safe refusal) and rejected (harmful).
            chosen_logits = _logits(chosen_batch)
            chosen_logp = _sequence_logp(chosen_logits, chosen_batch["labels"])

            rejected_logits = _logits(rejected_batch)
            rejected_logp = _sequence_logp(rejected_logits, rejected_batch["labels"])

            loss = simdpo_forget_loss(chosen_logp, rejected_logp, cfg.beta)

            # --- Retain CE term (simdpo_retain only) ------------------------
            if needs_retain:
                try:
                    r_batch = next(retain_iter)  # type: ignore[arg-type]
                except StopIteration:
                    logger.warning(
                        "simdpo_unlearn: retain iterable exhausted mid-epoch %d; "
                        "ending epoch early.",
                        epoch,
                    )
                    break
                r_logits = _logits(r_batch)
                shift_logits = r_logits[..., :-1, :].contiguous()
                shift_labels = r_batch["labels"][..., 1:].contiguous()
                retain_loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss + cfg.retain_coeff * retain_loss

            loss.backward()
            opt.step()
            steps += 1
            total_steps += 1

        logger.info(
            "simdpo_unlearn: epoch %d/%d (%s) — %d steps.",
            epoch, cfg.epochs, cfg.variant, steps,
        )

    logger.info(
        "simdpo_unlearn: completed (%d total optimizer steps, variant=%s).",
        total_steps, cfg.variant,
    )
    return model


def make_simdpo_pairs(
    harmful_batches: List[Dict[str, torch.Tensor]],
    refusal_response: str,
    tokenizer,
    max_len: int = 256,
) -> List[Dict[str, Dict[str, torch.Tensor]]]:
    """Utility: construct SimDPO pair batches from raw harmful batches.

    Takes harmful batches (each with ``input_ids`` / ``attention_mask`` /
    ``labels`` for a harmful completion) and a fixed safe refusal string,
    and produces paired batches ready for :func:`simdpo_unlearn`.

    The safe refusal is tokenized and used as the ``chosen`` completion for
    every pair.  The original harmful batch becomes the ``rejected`` completion.

    Args:
        harmful_batches: list of single-example harmful batches.
        refusal_response: the safe refusal text (e.g. ``"I'm sorry, I cannot
            assist with that request."``).
        tokenizer: HuggingFace tokenizer.
        max_len: max token length for the refusal encoding.

    Returns:
        List of ``{"chosen": batch, "rejected": batch}`` dicts, one per
        harmful batch.
    """
    SAFE_REFUSAL = refusal_response.strip()
    pairs = []
    for harm_batch in harmful_batches:
        # Decode the prompt portion from the harmful batch (labels == -100 marks
        # the prompt; we recover it by finding the first non-masked token).
        input_ids = harm_batch["input_ids"][0]
        labels = harm_batch["labels"][0]
        prompt_mask = labels == -100
        prompt_ids = input_ids[prompt_mask]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        # Build chosen = prompt + safe refusal.
        chosen_text = prompt_text + " " + SAFE_REFUSAL
        chosen_enc = tokenizer(
            chosen_text,
            return_tensors="pt",
            max_length=max_len,
            truncation=True,
            padding="max_length",
        )
        # BUGFIX (b): the prompt boundary must be measured against the SAME
        # encoding as ``chosen_text``.  ``len(prompt_ids)`` was derived from the
        # harmful batch's tokenization (different settings / special tokens), so
        # it could mask the wrong span.  Re-encode the prompt with identical
        # settings (sans padding) and use its token count as the boundary.
        prompt_enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            max_length=max_len,
            truncation=True,
            add_special_tokens=True,
        )
        prompt_len = int(prompt_enc["input_ids"].shape[1])

        # Label only the refusal tokens (after the prompt).
        chosen_labels = chosen_enc["input_ids"].clone()
        chosen_labels[0, :prompt_len] = -100
        # BUGFIX (a): with padding="max_length", trailing PAD tokens otherwise
        # remain as CE targets (training the model to emit pad).  Mask every
        # non-attended position (pad) to -100 so only real response tokens are
        # supervised.
        chosen_labels[chosen_enc["attention_mask"] == 0] = -100
        chosen_batch = {
            "input_ids": chosen_enc["input_ids"].to(input_ids.device),
            "attention_mask": chosen_enc["attention_mask"].to(input_ids.device),
            "labels": chosen_labels.to(input_ids.device),
        }
        pairs.append({"chosen": chosen_batch, "rejected": harm_batch})
    return pairs


__all__ = [
    "SimDPOUnlearnConfig",
    "simdpo_unlearn",
    "simdpo_forget_loss",
    "make_simdpo_pairs",
]