"""
SaLoRA — Safety-Alignment Preserved Low-Rank Adaptation
(Li, Si, Backes, Zhang & Wang, ICLR 2025; arXiv:2501.01765).

GPU-Accelerated Implementation: Computes the safety subspace via incremental 
Gram matrix accumulation (X^T X) natively in VRAM. All SVD and Eigendecomposition 
math is performed on the GPU to maximize throughput on high-VRAM hardware.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback just in case tqdm isn't installed in the environment
    tqdm = lambda x, **kwargs: x  

logger = logging.getLogger(__name__)


@dataclass
class SaLoRAConfig:
    """Configuration for SaLoRA."""
    rank: int = 8
    safety_rank: int = 8
    strength: float = 1.0
    param_filter: List[str] = None  # type: ignore[assignment]
    n_iter: int = 7
    task_init: bool = True

    def __post_init__(self) -> None:
        if self.param_filter is None:
            self.param_filter = ["lora_A"]


# ---------------------------------------------------------------------------
# Activation collection (Fully GPU-Accelerated)
# ---------------------------------------------------------------------------

class _ActivationRecorder:
    """Forward-hook helper that incrementally accumulates X^T X natively in VRAM."""

    def __init__(self) -> None:
        self.covs: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, inputs, output) -> None:  # noqa: ANN001
            x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            if not isinstance(x, torch.Tensor):
                return
            
            # Keep flat and covariance matrix strictly on the GPU
            flat = x.detach().reshape(-1, x.shape[-1]).float()
            cov = flat.T @ flat
            
            if name not in self.covs:
                self.covs[name] = cov
            else:
                self.covs[name] += cov

        return hook

    def attach(self, modules: Dict[str, nn.Module]) -> None:
        for name, mod in modules.items():
            self._handles.append(mod.register_forward_hook(self._make_hook(name)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _target_linears(
    model: nn.Module, param_filter: Iterable[str]
) -> Dict[str, nn.Linear]:
    """Find the *base* ``nn.Linear`` layers that a LoRA adapter wraps."""
    tokens = list(param_filter)
    out: Dict[str, nn.Linear] = {}
    
    for name, mod in model.named_modules():
        if getattr(mod, "lora_A", None) is None:
            continue
        base = getattr(mod, "base_layer", None)
        if isinstance(base, nn.Linear):
            out[f"{name}.base_layer"] = base
            
    if out:
        return out
        
    return {
        n: m
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and (not tokens or any(t in n for t in tokens))
    }


# ---------------------------------------------------------------------------
# Component 1: fixed safety module  C = I - V V.T
# ---------------------------------------------------------------------------

def compute_safety_subspace(
    aligned: nn.Module,
    base: Optional[nn.Module] = None,
    rank: int = 8,
    param_filter: Optional[List[str]] = None,
    *,
    safety_inputs: Optional[Iterable[dict]] = None,
    safety_rank: Optional[int] = None,
    n_iter: int = 7,
) -> Dict[str, torch.Tensor]:
    """Build SaLoRA's per-layer fixed safety module ``C = I - V V.T``."""
    rs = safety_rank if safety_rank is not None else rank
    pf = param_filter or ["lora_A"]

    if safety_inputs is None:
        logger.warning(
            "salora: compute_safety_subspace called without `safety_inputs`; "
            "falling back to the legacy weight-delta SVD."
        )
        return _legacy_weight_delta_subspace(aligned, base, rs, pf)

    modules = _target_linears(aligned, pf)
    if not modules:
        logger.warning("salora: no target linear layers found.")
        return {}

    recorder = _ActivationRecorder()
    recorder.attach(modules)
    device = next(aligned.parameters()).device
    aligned.eval()
    
    try:
        with torch.no_grad():
            for batch in tqdm(safety_inputs, desc="salora: accumulating gram matrices", leave=False):
                kwargs = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                aligned(**kwargs)
    finally:
        recorder.detach()

    covs = recorder.covs
    safety: Dict[str, torch.Tensor] = {}
    
    for name, mod in tqdm(modules.items(), desc="salora: computing subspaces (gpu eigh)", leave=False):
        cov_gpu = covs.get(name)
        if cov_gpu is None:
            continue
            
        # Keep everything in VRAM for fast eigendecomposition
        W = mod.weight.detach().float()  # (out_dim, in_dim) on GPU
        
        # S = W @ (X^T X) @ W^T
        S = torch.matmul(W, torch.matmul(cov_gpu, W.T))
        
        q = max(1, min(rs, min(S.shape) - 1 if min(S.shape) > 1 else 1))
        try:
            # Native GPU Eigendecomposition (cuSOLVER)
            _, eigvecs = torch.linalg.eigh(S)
            V = eigvecs[:, -q:]  # top q right singular vectors
        except RuntimeError as e:
            logger.warning("salora: eigh failed for %s (%s); skipping.", name, e)
            continue
            
        # Projector C computed entirely on GPU
        C = torch.eye(V.shape[0], dtype=V.dtype, device=W.device) - V @ V.T
        
        # Store on CPU to prevent dict from hoarding 50GB+ of VRAM across all layers.
        # It will be pushed back to VRAM exactly when needed by _install_safety_module.
        safety[name] = C.cpu()
        
        # Aggressive GPU garbage collection for large intermediate tensors
        del S, W, cov_gpu, eigvecs, V, C
        
    logger.info(
        "salora: built activation-derived safety module for %d layers (rs=%d).",
        len(safety), rs,
    )
    return safety


def _legacy_weight_delta_subspace(
    aligned: nn.Module,
    base: Optional[nn.Module],
    rank: int,
    param_filter: List[str],
) -> Dict[str, torch.Tensor]:
    """Backward-compatibility only: the old (SafeLoRA-style, not SaLoRA) path."""
    if base is None:
        return {}
    sub: Dict[str, torch.Tensor] = {}
    base_sd = base.state_dict()
    aligned_sd = aligned.state_dict()
    for name, w in aligned_sd.items():
        if name not in base_sd or w.dim() != 2:
            continue
        if param_filter and not any(s in name for s in param_filter):
            continue
        # Math remains on GPU if tensors are on GPU
        delta = w.float() - base_sd[name].float()
        try:
            _, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        except RuntimeError as e:
            logger.warning("salora: SVD failed for %s (%s); skipping.", name, e)
            continue
        k = max(1, min(rank, S.numel()))
        sub[name] = Vh[:k, :].T
    return sub


# ---------------------------------------------------------------------------
# Component 2: task-specific initialisation of the trainable LoRA params
# ---------------------------------------------------------------------------

def task_specific_init(
    model: nn.Module,
    config: Optional[SaLoRAConfig] = None,
) -> int:
    """Apply SaLoRA's task-specific GPU-accelerated SVD initialisation."""
    cfg = config or SaLoRAConfig()
    if not cfg.task_init:
        return 0

    inited = 0
    named = dict(model.named_modules())
    for name, mod in tqdm(named.items(), desc="salora: task-specific init (gpu svd)", leave=False):
        lora_A = getattr(mod, "lora_A", None)
        lora_B = getattr(mod, "lora_B", None)
        base = getattr(mod, "base_layer", None)
        
        if lora_A is None or lora_B is None or base is None:
            continue
            
        weight = getattr(base, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
            
        # Execute SVD natively on the GPU
        W = weight.detach().float()
        r = cfg.rank
        q = max(1, min(r, min(W.shape) - 1 if min(W.shape) > 1 else 1))
        
        try:
            U2, S2, V2 = torch.svd_lowrank(W, q=q, niter=cfg.n_iter)
        except RuntimeError as e:
            logger.warning("salora: task-init svd failed for %s (%s).", name, e)
            continue
            
        sqrtS = torch.diag(torch.sqrt(S2.clamp_min(0.0)))
        B_init = U2 @ sqrtS                    # (out_dim, q)
        A_init = sqrtS @ V2.T                  # (q, in_dim)

        def _assign(md, value: torch.Tensor) -> bool:
            items = md.values() if hasattr(md, "values") else [md]
            ok = False
            for sub in items:
                w = getattr(sub, "weight", None)
                if w is None:
                    continue
                with torch.no_grad():
                    if w.shape == value.shape:
                        w.copy_(value.to(w.dtype))
                        ok = True
                    elif w.shape == value.T.shape:
                        w.copy_(value.T.to(w.dtype))
                        ok = True
            return ok

        if _assign(lora_A, A_init) and _assign(lora_B, B_init):
            inited += 1
            
        # Clean up large GPU tensors
        del W, U2, S2, V2, B_init, A_init
            
    logger.info("salora: task-specific init applied to %d LoRA layers.", inited)
    return inited


# ---------------------------------------------------------------------------
# Forward-pass installation of the fixed safety module C
# ---------------------------------------------------------------------------

def _install_safety_module(
    model: nn.Module,
    safety_subspace: Dict[str, torch.Tensor],
    strength: float,
) -> int:
    """Wrap each LoRA layer's forward so its adapter output is projected by C."""
    installed = 0
    named = dict(model.named_modules())

    def _make_hook(peft_mod: nn.Module) -> Callable:
        def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
            C_buf = getattr(peft_mod, "lora_C", None)
            if C_buf is None:
                return output
            try:
                return output @ C_buf.to(output.dtype).to(output.device).T
            except RuntimeError:
                return output
        return hook

    for name, mod in tqdm(named.items(), desc="salora: installing hooks", leave=False):
        C = safety_subspace.get(name)
        if C is None:
            for key, val in safety_subspace.items():
                if key in name or name in key:
                    C = val
                    break
        if C is None:
            continue
        
        lora_B = getattr(mod, "lora_B", None)
        if lora_B is None:
            continue
            
        if getattr(mod, "_salora_installed", False):
            continue

        Cf = C.detach().clone()
        if strength != 1.0:
            I = torch.eye(Cf.shape[0], dtype=Cf.dtype)
            Cf = I - strength * (I - Cf)
            
        # Register as a buffer on the PEFT module (moves it back to the GPU context)
        mod.register_buffer("lora_C", Cf, persistent=False)

        items = lora_B.values() if hasattr(lora_B, "values") else [lora_B]
        hook_fn = _make_hook(mod)
        for sub in items:
            sub.register_forward_hook(hook_fn)

        mod._salora_installed = True  # type: ignore[attr-defined]
        installed += 1
        
    logger.info("salora: installed fixed safety module C in %d LoRA layers.", installed)
    return installed


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def project_lora_step(
    model: nn.Module,
    safety_subspace: Dict[str, torch.Tensor],
    config: Optional[SaLoRAConfig] = None,
    *,
    install_forward: bool = True,
    apply_task_init: bool = False,
) -> int:
    """Apply SaLoRA to ``model``."""
    cfg = config or SaLoRAConfig()

    if not safety_subspace:
        return 0

    sample = next(iter(safety_subspace.values()))
    is_safety_module = sample.dim() == 2 and sample.shape[0] == sample.shape[1]

    if is_safety_module:
        affected = 0
        if apply_task_init:
            affected += task_specific_init(model, cfg)
        if install_forward:
            affected += _install_safety_module(model, safety_subspace, cfg.strength)
        return affected

    logger.warning(
        "salora: project_lora_step received weight-delta singular vectors; "
        "running legacy per-step LoRA-grad projection."
    )
    projected = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not any(s in name for s in cfg.param_filter):
                continue
            if p.grad is None:
                continue
            V = safety_subspace.get(name)
            if V is None or V.shape[0] != p.grad.shape[-1]:
                continue
            V = V.to(p.grad.dtype).to(p.grad.device)
            parallel = (p.grad @ V) @ V.T
            p.grad.sub_(cfg.strength * parallel)
            projected += 1
    return projected


__all__ = [
    "SaLoRAConfig",
    "compute_safety_subspace",
    "task_specific_init",
    "project_lora_step",
]