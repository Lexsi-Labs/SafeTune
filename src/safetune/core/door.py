"""
DOOR / W-DOOR: Improving LLM Safety Alignment with Dual-Objective Optimization.
arXiv:2503.03710 — wicai24/DOOR-Alignment

Core idea:
    Jointly optimise two objectives during safety fine-tuning:
      1. **Refusal reinforcement** — cross-entropy loss pushing the model to
         produce refusal tokens on adversarial/harmful prompts.
      2. **Harmful capability unlearning** — negative log-likelihood loss on
         harmful completions to reduce their probability.
    W-DOOR adds a **KL divergence** term against a reference model to
    preserve general helpfulness / capability.

Integration with SafeTune:
    - ``build_door_dataset()`` creates paired (refused, harmful) training rows.
    - ``door_loss()`` implements the composite loss, callable from a TRL trainer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Refusal templates (extend as needed)
# ---------------------------------------------------------------------------

REFUSAL_TEMPLATES: List[str] = [
    "I'm sorry, but I can't help with that.",
    "I cannot assist with this request.",
    "That request goes against my guidelines. I won't help.",
    "I'm not able to provide that information.",
    "I must decline this request as it could be harmful.",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DOORConfig:
    """Configuration for DOOR / W-DOOR dual-objective safety alignment.

    Args:
        mode: ``"door"`` (refusal + unlearning only) or ``"w_door"``
            (adds KL divergence against reference model).
        refusal_weight: Weight for the refusal cross-entropy loss term.
        unlearn_weight: Weight for the harmful-unlearning loss term.
        kl_weight: Weight for KL divergence (W-DOOR only, ignored if
            ``mode="door"``).
        attack_types: Attack scenarios to simulate for refusal training
            (``"prefill"``, ``"multiturn"``).
        reference_model_path: Path to reference model for KL term
            (W-DOOR only).  If ``None``, KL term is skipped.
        refusal_templates: Custom refusal response templates.
            Defaults to ``REFUSAL_TEMPLATES``.
        max_harmful_pairs: Maximum harmful pairs to include in the dataset.
    """
    mode: str = "w_door"
    refusal_weight: float = 1.0
    unlearn_weight: float = 0.5
    kl_weight: float = 0.1
    attack_types: List[str] = field(default_factory=lambda: ["prefill", "multiturn"])
    reference_model_path: Optional[str] = None
    refusal_templates: List[str] = field(default_factory=lambda: list(REFUSAL_TEMPLATES))
    max_harmful_pairs: int = 10_000
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_door_config(**kwargs: Any) -> DOORConfig:
    """Build DOORConfig from a dict (e.g. YAML params)."""
    valid = {f.name for f in DOORConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return DOORConfig(**{k: v for k, v in kwargs.items() if k in valid})


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_door_dataset(
    harmful_rows: List[Dict[str, Any]],
    benign_rows: Optional[List[Dict[str, Any]]] = None,
    config: Optional[DOORConfig] = None,
) -> List[Dict[str, Any]]:
    """Create DOOR training pairs from harmful and (optionally) benign rows.

    Each output dict contains:
    - ``"prompt"``: the original harmful or adversarial prompt.
    - ``"refused"``: a canonical refusal response (positive target).
    - ``"harmful"``: the original harmful completion (negative / unlearn target).
    - ``"is_harmful"``: always ``True`` for DOOR pairs.
    - ``"attack_type"``: the attack scenario tag.

    Args:
        harmful_rows: List of ``{"prompt": ..., "response": ...}`` dicts
            representing harmful prompt-completion pairs.
        benign_rows: Optional benign pairs for helpfulness regularisation
            (used in W-DOOR to build KL reference targets).
        config: ``DOORConfig``; uses defaults if ``None``.

    Returns:
        List of augmented training dicts.
    """
    cfg = config or DOORConfig()
    pairs: List[Dict[str, Any]] = []
    templates = cfg.refusal_templates or REFUSAL_TEMPLATES

    for i, row in enumerate(harmful_rows[: cfg.max_harmful_pairs]):
        prompt   = str(row.get("prompt", row.get("question", "")))
        harmful  = str(row.get("response", row.get("answer", "")))
        refused  = templates[i % len(templates)]

        for attack in cfg.attack_types:
            # Simulate attack-type prefixes
            if attack == "prefill":
                adv_prompt = prompt + " Sure, here is"
            elif attack == "multiturn":
                adv_prompt = f"User: {prompt}\nAssistant: Definitely! "
            else:
                adv_prompt = prompt

            pairs.append({
                "prompt":      adv_prompt,
                "refused":     refused,
                "harmful":     harmful,
                "is_harmful":  True,
                "attack_type": attack,
            })

    # Add benign rows as W-DOOR KL anchors (no paired harmful completion)
    if benign_rows and cfg.mode == "w_door":
        for row in benign_rows:
            pairs.append({
                "prompt":      str(row.get("prompt", "")),
                "refused":     str(row.get("response", "")),
                "harmful":     "",
                "is_harmful":  False,
                "attack_type": "benign",
            })

    logger.info(
        "DOOR: built %d training pair(s) [mode=%s, attacks=%s].",
        len(pairs), cfg.mode, cfg.attack_types,
    )
    return pairs


# ---------------------------------------------------------------------------
# Dual-objective loss
# ---------------------------------------------------------------------------

def door_loss(
    model: Any,
    prompt_ids: Any,
    refused_ids: Any,
    harmful_ids: Optional[Any] = None,
    reference_model: Optional[Any] = None,
    config: Optional[DOORConfig] = None,
) -> Any:
    """Compute the DOOR / W-DOOR composite loss.

    Loss = refusal_weight * L_refusal
         + unlearn_weight * L_unlearn          (if harmful_ids provided)
         + kl_weight      * L_KL              (W-DOOR + reference_model)

    - **L_refusal**: Cross-entropy of model outputs on (prompt, refused) pairs.
    - **L_unlearn**: Negative log-likelihood on harmful completions
      (maximises loss → suppresses harmful generation).
    - **L_KL**: KL divergence between model and reference model log-probs on
      prompt (preserves helpfulness / capabilities).

    Args:
        model: The model being trained (must accept token ids).
        prompt_ids: Tokenised prompts ``(B, L)`` tensor.
        refused_ids: Tokenised refusal responses ``(B, L)`` tensor.
        harmful_ids: Tokenised harmful completions ``(B, L)`` tensor.  If
            ``None``, the unlearning term is skipped.
        reference_model: Reference model for KL (W-DOOR).  If ``None``,
            KL term is skipped even in ``w_door`` mode.
        config: ``DOORConfig``; uses defaults if ``None``.

    Returns:
        Scalar loss tensor.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise RuntimeError("door_loss requires PyTorch.")

    cfg = config or DOORConfig()

    def _seq_ce(logits: Any, targets: Any) -> Any:
        """Sequence-level cross-entropy (mean over non-padding tokens)."""
        B, L, V = logits.shape
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            targets[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    total_loss = torch.tensor(0.0, requires_grad=True)

    # 1) Refusal reinforcement
    try:
        ref_logits = model(
            input_ids=torch.cat([prompt_ids, refused_ids], dim=1)
        ).logits
        L_refusal = _seq_ce(ref_logits, torch.cat([prompt_ids, refused_ids], dim=1))
        total_loss = total_loss + cfg.refusal_weight * L_refusal
    except Exception as exc:
        logger.warning("DOOR: refusal loss failed — %s", exc)

    # 2) Harmful unlearning (maximise loss = minimise harmful generation prob)
    if harmful_ids is not None:
        try:
            harm_logits = model(
                input_ids=torch.cat([prompt_ids, harmful_ids], dim=1)
            ).logits
            L_unlearn = -_seq_ce(harm_logits, torch.cat([prompt_ids, harmful_ids], dim=1))
            total_loss = total_loss + cfg.unlearn_weight * L_unlearn
        except Exception as exc:
            logger.warning("DOOR: unlearning loss failed — %s", exc)

    # 3) KL divergence (W-DOOR)
    if cfg.mode == "w_door" and reference_model is not None:
        try:
            with torch.no_grad():
                ref_out = reference_model(input_ids=prompt_ids).logits
            curr_out = model(input_ids=prompt_ids).logits
            kl = F.kl_div(
                F.log_softmax(curr_out, dim=-1),
                F.softmax(ref_out, dim=-1),
                reduction="batchmean",
            )
            total_loss = total_loss + cfg.kl_weight * kl
        except Exception as exc:
            logger.warning("DOOR: KL loss failed — %s", exc)

    return total_loss


# ---------------------------------------------------------------------------
# Tokenisation helpers (thin wrappers for use inside DOORCallback)
# ---------------------------------------------------------------------------

def tokenize_door_batch(
    batch: List[Dict[str, Any]],
    tokenizer: Any,
    max_length: int = 512,
) -> Dict[str, Any]:
    """Tokenise a DOOR batch into prompt / refused / harmful id tensors.

    Returns a dict with keys ``"prompt_ids"``, ``"refused_ids"``,
    ``"harmful_ids"`` (all padded to ``max_length``).
    """
    try:
        import torch
    except ImportError:
        return {}

    def _tok(texts: List[str]) -> Any:
        enc = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"]

    prompts   = [r["prompt"]  for r in batch]
    refused   = [r["refused"] for r in batch]
    harmful   = [r.get("harmful", "") for r in batch]

    return {
        "prompt_ids":  _tok(prompts),
        "refused_ids": _tok(refused),
        "harmful_ids": _tok(harmful) if any(h for h in harmful) else None,
    }
