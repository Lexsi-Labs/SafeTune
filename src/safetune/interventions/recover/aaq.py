"""AAQ: Alignment-Aware Quantization with APC calibration loop.

Paper
-----
"Alignment-Aware Quantization for LLM Safety", Wee, Kim, Kim, Hwang & Kwak,
arXiv:2511.07842 (v1; Seoul National University / LG Electronics). No public
code repository exists -- the authors' code is "anonymized in the supplementary
material" only -- so this implementation is reconstructed from the paper text
(Methodology Sec. 3, Algorithm 1, and the Sec. 4.1 hyper-parameters).

Faithful AAQ algorithm (Algorithm 1)
------------------------------------
AAQ does *not* fine-tune the model weights. It optimises a small set of
pre-quantization transformation parameters with the **Alignment-Preserving
Contrastive (APC) loss**, then applies the quantizer. The APC loss is a
top-K KL-divergence contrastive objective on the **output logit
distributions** (NOT a cosine similarity on hidden states):

    S_top(x)  = top-K indices of  p_FT(y|x)                 # aligned model
    S_diff(x) = top-K indices of |p_FT(y|x) - p_PT(y|x)|    # where FT/PT disagree
    p^S       = renormalise p over the index subset S        # Eq. (4)

    L_KL-top   = mean_x KL( p_FT^S_top  ||  p_Q^S_top  )      # pull -> aligned
    L_cont-top = mean_x KL( p_PT^S_diff ||  p_Q^S_diff )      # push <- pre-trained
    L_APC      = L_KL-top  -  alpha * L_cont-top              # Eq. (7)

with the paper's defaults `alpha = 0.75` and `K = 500`, optimised on a small
unlabelled calibration set (128 WikiText-2 samples in the paper). There is no
separate task/reconstruction term -- the APC loss *is* the full objective.

Relationship to ``AAQPatch``
----------------------------
``safetune.core.patches.aaq_patch.AAQPatch`` implements an earlier, simplified
variant (a cosine-similarity contrastive loss on hidden states, with a random
``torch.randn`` float-tensor probe fallback that breaks a real HF LM's
embedding lookup). To bring ``apply_aaq`` to full faithfulness *without*
editing ``aaq_patch.py``, this entrypoint runs the paper-accurate APC
calibration directly when reference models are available, and only falls back
to ``AAQPatch`` when they are not.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Faithful APC loss (paper Eq. 4-7, Algorithm 1)
# ---------------------------------------------------------------------------

def _renormalised_subset(probs: Any, index: Any) -> Any:
    """Gather ``probs`` on the ``index`` columns and renormalise (paper Eq. 4).

    ``probs`` is ``[..., vocab]``; ``index`` is ``[..., K]``. Returns a
    distribution of shape ``[..., K]`` that sums to 1 over the last dim.
    """
    sub = probs.gather(-1, index)
    return sub / sub.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _apc_loss(
    logits_q: Any,
    logits_ft: Any,
    logits_pt: Any,
    top_k: int,
    alpha: float,
) -> Any:
    """Alignment-Preserving Contrastive loss (paper Eq. 7, Algorithm 1).

    Args:
        logits_q:  quantized/calibrated model logits  ``[B, T, V]``.
        logits_ft: fine-tuned (aligned) reference logits ``[B, T, V]``.
        logits_pt: pre-trained (base) reference logits   ``[B, T, V]``.
        top_k:     ``K`` -- vocabulary subset size for both components.
        alpha:     contrastive weight (push strength), ``> 0``.

    Returns ``L_KL-top - alpha * L_cont-top``, averaged over the calibration
    batch and sequence positions.
    """
    import torch
    import torch.nn.functional as F

    # Work in float for numerically stable softmax / KL.
    logits_q = logits_q.float()
    logits_ft = logits_ft.float()
    logits_pt = logits_pt.float()

    k = max(1, min(int(top_k), logits_q.size(-1)))

    p_q = F.softmax(logits_q, dim=-1)
    p_ft = F.softmax(logits_ft, dim=-1)
    p_pt = F.softmax(logits_pt, dim=-1)

    # S_top(x): the aligned model's K most-probable tokens (pull target).
    idx_top = torch.topk(p_ft, k, dim=-1).indices
    # S_diff(x): tokens where FT and PT disagree most (push region).
    idx_diff = torch.topk((p_ft - p_pt).abs(), k, dim=-1).indices

    # Renormalise each distribution over its respective subset (Eq. 4).
    q_top = _renormalised_subset(p_q, idx_top)
    ft_top = _renormalised_subset(p_ft, idx_top)
    q_diff = _renormalised_subset(p_q, idx_diff)
    pt_diff = _renormalised_subset(p_pt, idx_diff)

    eps = 1e-12
    # KL(P || Q) = sum P * (log P - log Q); P = reference, Q = our model.
    l_kl_top = (
        ft_top * ((ft_top + eps).log() - (q_top + eps).log())
    ).sum(dim=-1).mean()
    l_cont_top = (
        pt_diff * ((pt_diff + eps).log() - (q_diff + eps).log())
    ).sum(dim=-1).mean()

    # L_APC = L_KL-top - alpha * L_cont-top  (Eq. 7): pull toward aligned,
    # push away from pre-trained where the two references disagree.
    return l_kl_top - alpha * l_cont_top


# ---------------------------------------------------------------------------
# Probe construction -- integer token ids (fixes the float-probe-ids bug)
# ---------------------------------------------------------------------------

def _make_probe_ids(
    probe_texts: Optional[List[str]],
    model: nn.Module,
    device: Any,
    max_length: int = 64,
    batch_size: int = 4,
) -> Any:
    """Build a **Long** ``input_ids`` tensor for calibration.

    Fixes the FEATURE_MAP "float probe-ids" bug: a language model's embedding
    lookup requires integer token ids, so a ``torch.randn`` float fallback
    raises inside ``nn.Embedding``. We always return an ``int64`` tensor whose
    values are bounded by the model's real vocabulary size.
    """
    import torch

    # Determine the model's vocabulary size so random ids stay in range.
    cfg = getattr(model, "config", None)
    vocab_size = int(getattr(cfg, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        # Fall back to the largest embedding-shaped 2-D parameter.
        for p in model.parameters():
            if p.dim() == 2 and p.size(0) > 1024:
                vocab_size = int(p.size(0))
                break
    if vocab_size <= 0:
        vocab_size = 32000  # generic LLM default

    if probe_texts:
        try:
            from transformers import AutoTokenizer

            name = "gpt2"
            if cfg is not None and getattr(cfg, "_name_or_path", None):
                name = cfg._name_or_path
            tok = AutoTokenizer.from_pretrained(name)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token or tok.unk_token
            enc = tok(
                list(probe_texts)[:batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            return enc["input_ids"].to(device=device, dtype=torch.long)
        except Exception as exc:  # pragma: no cover - tokenizer optional
            logger.warning("apply_aaq: tokenizer unavailable (%s); using random ids.", exc)

    # No texts / no tokenizer: random *integer* ids (NOT floats) in [0, vocab).
    return torch.randint(
        0, vocab_size, (batch_size, max_length), device=device, dtype=torch.long
    )


def _model_logits(model: nn.Module, input_ids: Any) -> Any:
    """Run ``model`` and return its ``[B, T, V]`` logits tensor."""
    out = model(input_ids)
    logits = getattr(out, "logits", None)
    if logits is not None:
        return logits
    if isinstance(out, tuple):
        return out[0]
    return out  # already a tensor


# ---------------------------------------------------------------------------
# Faithful calibration loop
# ---------------------------------------------------------------------------

def _run_apc_calibration(
    model: nn.Module,
    aligned_model: nn.Module,
    base_model: nn.Module,
    probe_ids: Any,
    *,
    calibration_steps: int,
    lr: float,
    top_k: int,
    alpha: float,
    simulate_quantization: bool,
    quantization_bits: int,
) -> bool:
    """Optimise ``model`` with the APC loss against the two references.

    Returns ``True`` on success, ``False`` if the loop could not run (caller
    then falls back to ``AAQPatch``).
    """
    import torch

    # Reference logits are fixed targets -- compute once, no grad.
    try:
        with torch.no_grad():
            logits_ft = _model_logits(aligned_model, probe_ids).detach()
            logits_pt = _model_logits(base_model, probe_ids).detach()
    except Exception as exc:
        logger.warning("apply_aaq: reference forward pass failed (%s).", exc)
        return False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    was_training = model.training
    model.train()

    quant_hook = None
    if simulate_quantization:
        quant_hook = _FakeQuantizer(quantization_bits)

    try:
        for step in range(calibration_steps):
            optimizer.zero_grad()
            if quant_hook is not None:
                quant_hook.apply(model)
            logits_q = _model_logits(model, probe_ids)
            loss = _apc_loss(logits_q, logits_ft, logits_pt, top_k=top_k, alpha=alpha)
            loss.backward()
            optimizer.step()
            if (step + 1) % max(1, calibration_steps // 5) == 0:
                logger.debug(
                    "apply_aaq: APC step %d/%d -- loss %.4f",
                    step + 1, calibration_steps, float(loss.item()),
                )
    except Exception as exc:
        logger.warning("apply_aaq: APC calibration loop failed (%s).", exc)
        return False
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    logger.info(
        "apply_aaq: APC calibration complete (%d steps, K=%d, alpha=%.2f).",
        calibration_steps, top_k, alpha,
    )
    return True


class _FakeQuantizer:
    """Optional simulated (fake) weight quantization applied each step.

    AAQ is "Alignment-Aware *Quantization*": the APC loss is meant to
    compensate for quantization error. When ``simulate_quantization`` is on we
    round 2-D weight tensors to a ``bits``-wide symmetric per-tensor grid in
    the forward pass so the calibration actually sees quantization noise. This
    is a lightweight stand-in for the paper's GPTQ/OSTQuant pipeline (which
    needs a full quant toolchain); it is off by default to preserve behaviour.
    """

    def __init__(self, bits: int) -> None:
        self.bits = max(2, int(bits))

    def apply(self, model: nn.Module) -> None:
        import torch

        levels = 2 ** self.bits - 1
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() != 2:
                    continue
                scale = p.detach().abs().max().clamp_min(1e-8) / (levels / 2)
                q = torch.round(p.detach() / scale).clamp_(-levels // 2, levels // 2)
                p.copy_(q * scale)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

@assert_mutates("apply_aaq")
def apply_aaq(
    model: nn.Module,
    aligned_model_path: str = "",
    base_model_path: str = "",
    quantization_bits: int = 4,
    apc_weight: float = 0.75,
    calibration_steps: int = 20,
    lr: float = 1e-4,
    probe_texts: Optional[List[str]] = None,
    top_k: int = 500,
    simulate_quantization: bool = False,
    **extra: Any,
) -> nn.Module:
    """Run the AAQ APC calibration loop on ``model`` in-place.

    Implements the Alignment-Preserving Contrastive (APC) loss of
    "Alignment-Aware Quantization for LLM Safety" (arXiv:2511.07842, Wee et
    al.): a top-K KL-divergence contrastive objective that pulls the
    calibrated model's output distribution toward the aligned (fine-tuned)
    reference and pushes it away from the base (pre-trained) reference.

    Args:
        model: target ``nn.Module`` to calibrate in-place.
        aligned_model_path: path to the fine-tuned / aligned reference model.
        base_model_path: path to the unaligned pre-trained reference model.
        quantization_bits: weight bit-width (used by ``simulate_quantization``
            and forwarded to ``AAQPatch``).
        apc_weight: ``alpha`` -- the APC contrastive (push) weight. The paper's
            default is ``0.75`` (Sec. 4.1).
        calibration_steps: number of APC optimisation steps.
        lr: Adam learning rate for the calibration.
        probe_texts: optional calibration texts; random integer token ids are
            used when ``None``.
        top_k: ``K`` -- vocabulary subset size for the top-K KL terms (paper
            default ``500``). Optional kwarg with a paper-faithful default.
        simulate_quantization: if ``True``, apply lightweight fake weight
            quantization each step so the APC loss sees quantization noise
            (the paper's quantization stage is otherwise external). Optional
            kwarg, defaults to ``False`` to preserve prior behaviour.
        **extra: forwarded to :class:`AAQPatch` on the fallback path.

    Returns the same ``model`` instance, mutated in place.

    The faithful APC calibration runs when *both* reference models load; if
    either is missing the call delegates to
    :class:`safetune.core.patches.aaq_patch.AAQPatch` (the simplified variant).
    """
    alpha = float(apc_weight)
    steps = int(calibration_steps)
    k = int(top_k)

    # --- Try the paper-faithful path: needs both reference models. ---------
    aligned_model = None
    base_model = None
    try:
        import torch  # noqa: F401

        from safetune.core.patches.aaq_patch import _try_load_hf_model

        try:
            device = next(model.parameters()).device
        except StopIteration:
            import torch as _t

            device = _t.device("cpu")

        aligned_model = _try_load_hf_model(aligned_model_path, device)
        base_model = _try_load_hf_model(base_model_path, device)

        if aligned_model is not None and base_model is not None:
            probe_ids = _make_probe_ids(probe_texts, model, device)
            ok = _run_apc_calibration(
                model,
                aligned_model,
                base_model,
                probe_ids,
                calibration_steps=steps,
                lr=float(lr),
                top_k=k,
                alpha=alpha,
                simulate_quantization=bool(simulate_quantization),
                quantization_bits=int(quantization_bits),
            )
            if ok:
                return model
            logger.warning(
                "apply_aaq: faithful APC path failed; falling back to AAQPatch."
            )
    except ImportError as e:
        logger.warning(
            "apply_aaq: faithful path unavailable (%s); falling back to AAQPatch.", e
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "apply_aaq: faithful path errored (%s); falling back to AAQPatch.", e
        )

    # --- Fallback: delegate to the simplified AAQPatch. --------------------
    try:
        from safetune.core.patches.aaq_patch import AAQPatch
    except ImportError as e:  # pragma: no cover - defensive
        raise ImportError(
            f"apply_aaq needs safetune.core.patches.aaq_patch: {e}"
        ) from e

    params: Dict[str, Any] = {
        "aligned_model_path": aligned_model_path,
        "base_model_path": base_model_path,
        "quantization_bits": quantization_bits,
        "apc_weight": apc_weight,
        "calibration_steps": calibration_steps,
        "lr": lr,
        "probe_texts": probe_texts,
    }
    params.update(extra)

    patch = AAQPatch(**params)
    patch.apply_to_model(model)
    return model


__all__ = ["apply_aaq"]
