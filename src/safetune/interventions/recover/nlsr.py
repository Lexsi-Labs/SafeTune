"""Model-merging and safety recovery primitives (LSSF, SOMF, Task Arithmetic, NLSR)."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


# =============================================================================
# 1. LOW-RANK SAFETY SUBSPACE FUSION (LSSF)
# =============================================================================

def _safety_critical_rank(S: torch.Tensor, eta: float, max_rank: int) -> int:
    """Smallest rank whose cumulative singular-value entropy retains ``eta``."""
    n = int(S.numel())
    if n == 0:
        return 0
    cap = max(1, min(max_rank, n))
    energy = (S.double() ** 2)
    total = energy.sum()
    if total <= 0:
        return cap
    p = energy / total
    safe_p = torch.where(p > 0, p, torch.ones_like(p))
    terms = -p * torch.log(safe_p)
    H_full = terms.sum()
    if H_full <= 0:
        return min(1, cap)
    H_cum = torch.cumsum(terms, dim=0)
    ratio = H_cum / H_full
    meets = (ratio >= eta).nonzero(as_tuple=False)
    r = int(meets[0].item()) + 1 if meets.numel() > 0 else n
    return max(1, min(r, cap))


def _weighted_basis(U: torch.Tensor, S: torch.Tensor, r: int, weight_max: float) -> torch.Tensor:
    """Top-r left-singular basis with columns scaled per paper Eqs. 10-11."""
    Ur = U[:, :r]
    if weight_max <= 1.0:
        return Ur
    if r == 1:
        # Edge case fix: Scale by max weight if there is only 1 dominant direction.
        return Ur * weight_max
        
    Sr = S[:r]
    s1, sr = Sr[0], Sr[-1]
    span = (s1 - sr)
    if span <= 0:
        return Ur
    alpha_i = 1.0 + (weight_max - 1.0) * (Sr - sr) / span  # Eq. 11
    return Ur * alpha_i.clamp_min(0.0).to(Ur.dtype)


@assert_mutates("apply_lssf")
def apply_lssf(
    finetuned: nn.Module,
    base: nn.Module,
    aligned: nn.Module,
    alpha: float = 1.0,
    rank: int = 8,
    min_param_dim: int = 4,
    skip_param_substrings: Optional[list] = None,
    eta: Optional[float] = None,
    weight_max: float = 1.0,
    subspace_basis: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    """Apply Low-Rank Safety Subspace Fusion in place to ``finetuned``."""
    skip = list(skip_param_substrings or ["embed_tokens", "lm_head", "norm"])
    subspace_basis = subspace_basis or {}

    base_sd = base.state_dict()
    aligned_sd = aligned.state_dict()
    ft_sd = finetuned.state_dict()

    edited = 0
    skipped = 0
    rank_sum = 0
    with torch.no_grad():
        for name, ft_w in ft_sd.items():
            if any(s in name for s in skip) or name not in base_sd or name not in aligned_sd:
                skipped += 1
                continue
            if ft_w.dim() != 2 or min(ft_w.shape) < min_param_dim:
                skipped += 1
                continue

            # Build basis. Compute SVD on CPU to prevent VRAM spikes.
            if name in subspace_basis:
                U = subspace_basis[name].to(ft_w.device).float()
                S = None
            else:
                try:
                    delta_cpu = (
                        aligned_sd[name].to(torch.float32).cpu() - 
                        base_sd[name].to(torch.float32).cpu()
                    )
                    U_cpu, S_cpu, _ = torch.linalg.svd(delta_cpu, full_matrices=False)
                    U = U_cpu.to(ft_w.device)
                    S = S_cpu.to(ft_w.device)
                except RuntimeError as e:
                    logger.warning("LSSF: SVD failed for %s (%s); skipping.", name, e)
                    continue

            if eta is not None and S is not None:
                r = _safety_critical_rank(S, eta=float(eta), max_rank=rank)
            else:
                r = max(1, min(rank, U.shape[1]))
                
            if r == 0:
                skipped += 1
                continue

            Ur = _weighted_basis(U, S, r, weight_max) if S is not None else U[:, :r]

            # Re-calculate delta on GPU strictly for projection
            delta = (aligned_sd[name].to(ft_w.dtype) - base_sd[name].to(ft_w.dtype)).to(ft_w.device).float()
            proj = Ur @ (Ur.transpose(0, 1) @ delta)
            ft_w.add_((alpha * proj).to(ft_w.dtype))
            
            edited += 1
            rank_sum += r

    avg_rank = (rank_sum / edited) if edited else 0.0
    logger.info("LSSF: edited %d 2-D params (skipped %d), alpha=%.2f eta=%s avg_rank=%.1f.",
                edited, skipped, alpha, eta, avg_rank)
    return finetuned


# =============================================================================
# 2. SOMF, TASK ARITHMETIC, AND PRE-POST MERGING
# =============================================================================

def _resolve_pre_model(pre_model: Union[nn.Module, str]) -> nn.Module:
    if isinstance(pre_model, nn.Module):
        return pre_model
    if not isinstance(pre_model, str):
        raise TypeError(f"pre_model must be an nn.Module or string; got {type(pre_model)!r}")
    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]
    return AutoModelForCausalLM.from_pretrained(pre_model, torch_dtype="auto")


@assert_mutates("task_arithmetic")
def task_arithmetic(finetuned: nn.Module, base: nn.Module, aligned: nn.Module, alpha: float = 1.0) -> nn.Module:
    ft_sd, base_sd, aligned_sd = finetuned.state_dict(), base.state_dict(), aligned.state_dict()
    n_applied = 0
    with torch.no_grad():
        for name, ft_p in ft_sd.items():
            if name not in base_sd or name not in aligned_sd: continue
            task_vec = aligned_sd[name].to(ft_p.device, dtype=torch.float32) - base_sd[name].to(ft_p.device, dtype=torch.float32)
            updated = ft_p.to(torch.float32) + alpha * task_vec
            ft_p.copy_(updated.to(ft_p.dtype))
            n_applied += 1
    logger.info("task_arithmetic: applied safety vector to %d params", n_applied)
    return finetuned


def _somf_subspace_mask(task_vec: torch.Tensor, mask_threshold: float) -> torch.Tensor:
    if task_vec.numel() == 0: return torch.ones_like(task_vec, dtype=torch.bool)
    mag_f = task_vec.abs().float()
    MAX_QUANTILE_ELEMENTS = 10_000_000
    if mag_f.numel() > MAX_QUANTILE_ELEMENTS:
        indices = torch.randint(0, mag_f.numel(), (MAX_QUANTILE_ELEMENTS,), device=mag_f.device)
        thr = torch.quantile(mag_f.flatten()[indices], mask_threshold)
    else:
        thr = torch.quantile(mag_f, mask_threshold)
    return mag_f >= thr


@assert_mutates("somf_merge")
def somf_merge(finetuned: nn.Module, aligned: nn.Module, base: nn.Module, mask_threshold: float = 0.9, lam: float = 1.0, subspace_mask: dict | None = None) -> nn.Module:
    ft_sd, aligned_sd, base_sd = finetuned.state_dict(), aligned.state_dict(), base.state_dict()
    n_applied = 0
    with torch.no_grad():
        for name, ft_p in ft_sd.items():
            if name not in aligned_sd or name not in base_sd: continue
            ft_f, base_f, aligned_f = ft_p.to(torch.float32), base_sd[name].to(ft_p.device, dtype=torch.float32), aligned_sd[name].to(ft_p.device, dtype=torch.float32)
            task_vec = ft_f - base_f
            if task_vec.numel() == 0: continue
            mask = subspace_mask[name].to(ft_p.device, dtype=torch.bool) if subspace_mask and name in subspace_mask else _somf_subspace_mask(task_vec, mask_threshold)
            merged = aligned_f + lam * (task_vec * mask.to(torch.float32))
            ft_p.copy_(merged.to(ft_p.dtype))
            n_applied += 1
    logger.info("somf_merge: fused %d params", n_applied)
    return finetuned


@assert_mutates("apply_prepost_merge")
def apply_prepost_merge(model: nn.Module, pre_model: Union[nn.Module, str], alpha: float = 0.5, param_filter: Optional[Callable] = None, **extra) -> nn.Module:
    pre_sd = _resolve_pre_model(pre_model).state_dict()
    n_applied = 0
    with torch.no_grad():
        for name, post_p in model.named_parameters():
            if name not in pre_sd or (param_filter and not param_filter(name, post_p)): continue
            pre_p = pre_sd[name].to(post_p.device, dtype=torch.float32)
            merged = (1.0 - alpha) * post_p.to(torch.float32) + alpha * pre_p
            post_p.copy_(merged.to(post_p.dtype))
            n_applied += 1
    return model


# =============================================================================
# 3. NEURON-LEVEL SAFETY REALIGNMENT (NLSR)
# =============================================================================

def _load_state_dict(path: str) -> Dict[str, Any]:
    import torch as _torch
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file  # type: ignore[import-not-found]
        return dict(load_file(path, device="cpu"))
    raw = _torch.load(path, map_location="cpu", weights_only=True)
    return raw.get("model", raw) if isinstance(raw, dict) else raw


def _nlsr_stage3_transplant(
    model: nn.Module,
    donor: Dict[str, Any],
    region_mask: Optional[Dict[str, Any]],
    blend: float,
    tau: Optional[float],
) -> None:
    import torch as _torch
    import torch.nn.functional as _F

    transplanted, gated_out, total = 0, 0, 0
    with _torch.no_grad():
        for name, param in model.named_parameters():
            if name not in donor: continue
            total += 1
            donor_tensor = donor[name].to(device=param.device, dtype=param.dtype)
            
            if donor_tensor.shape != param.shape: continue

            mask = _torch.ones_like(param, dtype=_torch.bool)
            if region_mask is not None and name in region_mask:
                mask_tensor = region_mask[name].to(device=param.device).bool()
                if mask_tensor.shape == param.shape:
                    mask = mask_tensor

            if tau is not None:
                ft_region = param[mask].flatten().float()
                donor_region = donor_tensor[mask].flatten().float()
                
                if ft_region.numel() == 0:
                    gated_out += 1
                    continue
                
                # Apply the tau gate to ALL region sizes. A 1-element cosine is
                # degenerate, so treat its similarity as 1.0 (always >= tau) ->
                # gated out, matching the >=2 behavior instead of transplanting
                # single-element regions unconditionally.
                if ft_region.numel() >= 2:
                    cos_sim = _F.cosine_similarity(ft_region, donor_region, dim=0).item()
                else:
                    cos_sim = 1.0
                if cos_sim >= float(tau):
                    gated_out += 1
                    continue

            if blend >= 1.0:
                new_region = donor_tensor[mask]
            else:
                new_region = (1.0 - blend) * param[mask] + blend * donor_tensor[mask]
            
            # Safe, autograd-compliant in-place modification (replaces deprecated .data)
            param[mask] = new_region.to(param.dtype)
            transplanted += 1

    logger.info("apply_nlsr: transplanted %d/%d (gated out: %d)", transplanted, total, gated_out)


@assert_mutates("apply_nlsr")
def apply_nlsr(
    model: nn.Module,
    donor_param_path: Optional[str] = None,
    donor_map: Optional[Dict[str, Dict[int, float]]] = None,
    blend: float = 1.0,
    donor_state: Optional[Dict[str, Any]] = None,
    region_mask: Optional[Dict[str, Any]] = None,
    region_mask_path: Optional[str] = None,
    tau: Optional[float] = None,
    **extra: Any,
) -> nn.Module:
    """Transplant safety-reference neuron values into ``model`` using NLSR (stage 3)."""
    resolved_donor = dict(donor_state) if donor_state else (_load_state_dict(donor_param_path) if donor_param_path else None)
    
    resolved_mask = dict(region_mask) if region_mask else None
    if not resolved_mask and region_mask_path:
        import torch as _torch
        resolved_mask = dict(_torch.load(region_mask_path, map_location="cpu", weights_only=True))

    if resolved_donor is not None:
        _nlsr_stage3_transplant(model, donor=resolved_donor, region_mask=resolved_mask, blend=float(blend), tau=tau)
        return model

    from safetune.core.patches.nlsr_patch import NLSRPatch
    params = {"blend": blend, **extra}
    if donor_param_path: params["donor_param_path"] = donor_param_path
    if donor_map: params["donor_map"] = donor_map
    NLSRPatch(**params).apply_to_model(model)
    return model

__all__ = ["apply_nlsr"]