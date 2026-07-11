"""Model-merging primitives for training-free safety recovery."""
from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


def _resolve_pre_model(
    pre_model: Union[nn.Module, str],
) -> nn.Module:
    """Resolve *pre_model* to an ``nn.Module``.

    Accepts:
    * an already-constructed ``nn.Module`` (returned as-is),
    * a local filesystem path (directory or checkpoint file) or a
      Hugging Face model-id string — loaded via
      ``transformers.AutoModelForCausalLM.from_pretrained``.
    """
    if isinstance(pre_model, nn.Module):
        return pre_model

    if not isinstance(pre_model, str):
        raise TypeError(
            f"pre_model must be an nn.Module, a local path, or a HuggingFace "
            f"model-id string; got {type(pre_model)!r}"
        )

    try:
        from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "transformers is required to load pre_model from a path or HF id. "
            "Install it with: pip install transformers"
        ) from exc

    logger.info("apply_prepost_merge: loading pre_model from %r", pre_model)
    return AutoModelForCausalLM.from_pretrained(pre_model, torch_dtype="auto")


@assert_mutates("task_arithmetic")
def task_arithmetic(
    finetuned: nn.Module,
    base: nn.Module,
    aligned: nn.Module,
    alpha: float = 1.0,
) -> nn.Module:
    """Task-arithmetic safety recovery (Ilharco et al., 2023).

    Computes the safety task vector (aligned - base) and adds it to
    finetuned: theta_safe = theta_ft + alpha * (theta_aligned - theta_base).
    Mutates and returns ``finetuned`` in place.
    """
    ft_sd = finetuned.state_dict()
    base_sd = base.state_dict()
    aligned_sd = aligned.state_dict()
    n_applied = 0
    with torch.no_grad():
        for name, ft_p in ft_sd.items():
            if name not in base_sd or name not in aligned_sd:
                continue
            task_vec = aligned_sd[name].to(ft_p.device, dtype=torch.float32) - base_sd[name].to(
                ft_p.device, dtype=torch.float32
            )
            updated = ft_p.to(torch.float32) + alpha * task_vec
            ft_p.copy_(updated.to(ft_p.dtype))
            n_applied += 1
    logger.info("task_arithmetic: applied safety vector to %d params (alpha=%.3f)", n_applied, alpha)
    return finetuned


def _somf_subspace_mask(
    task_vec: torch.Tensor,
    mask_threshold: float,
) -> torch.Tensor:
    """Magnitude proxy for SOMF's learned safety-subspace mask.
    
    Optimized to use torch.quantile for efficient percentile cutoffs.
    Bypasses PyTorch's 16.7M element quantile limit via subsampling.
    """
    if task_vec.numel() == 0:
        return torch.ones_like(task_vec, dtype=torch.bool)
        
    mag = task_vec.abs()
    # torch.quantile requires float32 or float64
    mag_f = mag.float() if mag.dtype not in (torch.float32, torch.float64) else mag
    
    # PyTorch's torch.quantile crashes on tensors with > 16.7M elements.
    # We use 10M as a safe ceiling.
    MAX_QUANTILE_ELEMENTS = 10_000_000
    
    if mag_f.numel() > MAX_QUANTILE_ELEMENTS:
        # For massive matrices, a 10M random sample provides a statistically 
        # identical percentile threshold in a fraction of the time.
        flat_mag = mag_f.flatten()
        indices = torch.randint(0, flat_mag.numel(), (MAX_QUANTILE_ELEMENTS,), device=mag_f.device)
        sample_mag = flat_mag[indices]
        thr = torch.quantile(sample_mag, mask_threshold)
    else:
        thr = torch.quantile(mag_f, mask_threshold)
        
    return mag >= thr


@assert_mutates("somf_merge")
def somf_merge(
    finetuned: nn.Module,
    aligned: nn.Module,
    base: nn.Module,
    mask_threshold: float = 0.9,
    lam: float = 1.0,
    subspace_mask: dict | None = None,
) -> nn.Module:
    """Subspace-Oriented Model Fusion (SOMF), Yi et al., 2024 (arXiv:2405.09055).

    ⚠️ FIDELITY: SOMF's defining contribution is a **learned** Concrete/Gumbel
    probabilistic mask trained with a DPO safety objective. By default this
    function uses a **magnitude top-quantile heuristic** mask
    (:func:`_somf_subspace_mask`), which is NOT the paper's learned mask. For a
    faithful mask, train one via :func:`learn_somf_mask` and pass it as
    ``subspace_mask=``; otherwise treat the default as "SOMF-style heuristic
    fusion," not the published SOMF.
    """
    ft_sd = finetuned.state_dict()
    aligned_sd = aligned.state_dict()
    base_sd = base.state_dict()
    n_applied = 0
    with torch.no_grad():
        for name, ft_p in ft_sd.items():
            if name not in aligned_sd or name not in base_sd:
                continue
            ft_f = ft_p.to(torch.float32)
            base_f = base_sd[name].to(ft_p.device, dtype=torch.float32)
            aligned_f = aligned_sd[name].to(ft_p.device, dtype=torch.float32)
            
            task_vec = ft_f - base_f
            if task_vec.numel() == 0:
                continue
                
            if subspace_mask is not None and name in subspace_mask:
                mask = subspace_mask[name].to(ft_p.device, dtype=torch.bool)
            else:
                mask = _somf_subspace_mask(task_vec, mask_threshold)
                
            masked_delta = task_vec * mask.to(torch.float32)
            merged = aligned_f + lam * masked_delta
            ft_p.copy_(merged.to(ft_p.dtype))
            n_applied += 1
            
    logger.info(
        "somf_merge: fused %d params (mask_threshold=%.2f, lam=%.3f, mask=%s)",
        n_applied,
        mask_threshold,
        lam,
        "supplied" if subspace_mask is not None else "magnitude-proxy",
    )
    return finetuned


def learn_somf_mask(
    finetuned: nn.Module,
    aligned: nn.Module,
    base: nn.Module,
    preference_data: Iterable[Mapping],
    num_steps: int = 200,
    lr: float = 1e-2,
    temperature: float = 0.1,
    beta: float = 0.1,
    lam: float = 1.0,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Learn the SOMF subspace mask M via Concrete relaxation + DPO safety loss."""
    try:
        from torch.func import functional_call  # PyTorch >= 2.0
    except ImportError:
        try:
            from functorch import functional_call  # functorch fallback
        except ImportError as exc:
            raise RuntimeError(
                "learn_somf_mask requires torch.func.functional_call "
                "(PyTorch >= 2.0) or functorch."
            ) from exc

    torch.manual_seed(seed)
    dev = torch.device(device)

    ft_sd = finetuned.state_dict()
    aligned_sd = aligned.state_dict()
    base_sd = base.state_dict()

    delta: Dict[str, torch.Tensor] = {}
    log_alpha: Dict[str, torch.Tensor] = {}

    for name, ft_p in ft_sd.items():
        if name not in aligned_sd or name not in base_sd:
            continue
        if ft_p.dim() < 2:
            continue
        d = (
            ft_p.to(dev, dtype=torch.float32)
            - base_sd[name].to(dev, dtype=torch.float32)
        )
        if d.numel() == 0:
            continue
        delta[name] = d.detach()
        log_alpha[name] = torch.zeros_like(d, requires_grad=True)

    if not log_alpha:
        logger.warning(
            "learn_somf_mask: no 2-D parameters found in common across "
            "finetuned/aligned/base; returning an empty mask."
        )
        return {}

    aligned_params_dev: Dict[str, torch.Tensor] = {
        name: aligned_sd[name].to(dev, dtype=torch.float32).detach()
        for name in log_alpha
    }
    
    aligned_full_sd: Dict[str, torch.Tensor] = {}
    for name, p in aligned_sd.items():
        aligned_full_sd[name] = p.to(dev, dtype=torch.float32).detach()

    optimizer = torch.optim.Adam(list(log_alpha.values()), lr=lr)
    pref_list = list(preference_data)
    
    if not pref_list:
        logger.warning(
            "learn_somf_mask: preference_data is empty; "
            "returning all-ones mask (no training performed)."
        )
        return {name: torch.ones_like(d, dtype=torch.bool) for name, d in delta.items()}

    n_params = len(log_alpha)
    logger.info(
        "learn_somf_mask: training Concrete mask over %d params for %d steps "
        "(lr=%.3g, temp=%.3g, beta=%.3g, lam=%.3g, device=%s).",
        n_params, num_steps, lr, temperature, beta, lam, device,
    )

    pref_cycle = _cycle(pref_list)
    for step in range(num_steps):
        batch = next(pref_cycle)
        prompt_ids = batch["input_ids"].to(dev)
        chosen_ids = batch["chosen_ids"].to(dev)
        rejected_ids = batch["rejected_ids"].to(dev)

        optimizer.zero_grad(set_to_none=True)

        fused_sd: Dict[str, torch.Tensor] = dict(aligned_full_sd)
        for name, la in log_alpha.items():
            u = torch.zeros_like(la).uniform_().clamp(min=1e-6, max=1.0 - 1e-6)
            
            # Use standard logistic noise (Gumbel_1 - Gumbel_2) for binary Concrete distribution
            logistic_noise = torch.log(u) - torch.log(1.0 - u)
            m_soft = torch.sigmoid((la + logistic_noise) / temperature)
            fused_sd[name] = aligned_params_dev[name] + lam * m_soft * delta[name]

        # Standard DPO objective requires log probs from a reference model (aligned model)
        with torch.no_grad():
            ref_log_p_chosen = _somf_sequence_logprob(
                aligned, aligned_full_sd, prompt_ids, chosen_ids, functional_call
            )
            ref_log_p_rejected = _somf_sequence_logprob(
                aligned, aligned_full_sd, prompt_ids, rejected_ids, functional_call
            )

        # Log probs from the fused model
        log_p_chosen = _somf_sequence_logprob(
            aligned, fused_sd, prompt_ids, chosen_ids, functional_call
        )
        log_p_rejected = _somf_sequence_logprob(
            aligned, fused_sd, prompt_ids, rejected_ids, functional_call
        )
        
        margin_chosen = log_p_chosen - ref_log_p_chosen
        margin_rejected = log_p_rejected - ref_log_p_rejected

        # Final DPO loss over the margins
        loss = -F.logsigmoid(beta * (margin_chosen - margin_rejected)).mean()
        loss.backward()
        optimizer.step()

        if (step + 1) % max(1, num_steps // 5) == 0:
            logger.debug(
                "learn_somf_mask: step %d/%d  loss=%.4f",
                step + 1, num_steps, loss.item(),
            )

    mask: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, la in log_alpha.items():
            mask[name] = (la > 0).cpu()

    n_active = sum(int(m.sum().item()) for m in mask.values())
    n_total = sum(int(m.numel()) for m in mask.values())
    logger.info(
        "learn_somf_mask: mask learned — %d / %d coordinates active (%.1f%%).",
        n_active, n_total, 100.0 * n_active / max(1, n_total),
    )
    return mask


def _cycle(lst: list):
    """Infinite cyclic iterator over a list."""
    while True:
        for item in lst:
            yield item


def _somf_sequence_logprob(
    model: nn.Module,
    fused_sd: Dict[str, torch.Tensor],
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    functional_call,
) -> torch.Tensor:
    """Compute the sum of teacher-forced token log-probs for ``response_ids``."""
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)  # (1, T)
    prompt_len = prompt_ids.shape[1]
    resp_len = response_ids.shape[1]

    out = functional_call(model, fused_sd, (full_ids,))
    logits = out.logits if hasattr(out, "logits") else out[0]
    
    resp_logits = logits[:, prompt_len - 1: prompt_len + resp_len - 1, :]  # (1, resp_len, V)
    resp_targets = full_ids[:, prompt_len: prompt_len + resp_len]  # (1, resp_len)

    log_probs = F.log_softmax(resp_logits, dim=-1)  # (1, resp_len, V)
    tok_log_probs = log_probs.gather(
        dim=-1, index=resp_targets.unsqueeze(-1)
    ).squeeze(-1)  # (1, resp_len)

    return tok_log_probs.sum(dim=-1)


@assert_mutates("apply_prepost_merge")
def apply_prepost_merge(
    model: nn.Module,
    pre_model: Union[nn.Module, str],
    alpha: float = 0.5,
    param_filter: Optional[Callable] = None,
    **extra,
) -> nn.Module:
    """Restore safety by interpolating toward the pre-fine-tuning checkpoint."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    pre = _resolve_pre_model(pre_model)
    pre_sd = pre.state_dict()

    n_applied = 0
    n_skipped_filter = 0
    n_missing = 0
    with torch.no_grad():
        for name, post_p in model.named_parameters():
            if name not in pre_sd:
                n_missing += 1
                continue
            if param_filter is not None and not param_filter(name, post_p):
                n_skipped_filter += 1
                continue
            pre_p = pre_sd[name].to(post_p.device, dtype=torch.float32)
            post_f = post_p.to(torch.float32)
            merged = (1.0 - alpha) * post_f + alpha * pre_p
            post_p.copy_(merged.to(post_p.dtype))
            n_applied += 1

    logger.info(
        "apply_prepost_merge: interpolated %d params (alpha=%.3f, "
        "skipped_filter=%d, missing_in_pre=%d)",
        n_applied,
        alpha,
        n_skipped_filter,
        n_missing,
    )
    return model


__all__ = ["task_arithmetic", "somf_merge", "learn_somf_mask", "apply_prepost_merge"]