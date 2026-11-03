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
# Component 2: SVD initialisation of the trainable LoRA params (no task-feature projection)
# ---------------------------------------------------------------------------

def task_specific_init(
    model: nn.Module,
    config: Optional[SaLoRAConfig] = None,
) -> int:
    """Initialise ``A``, ``B`` from the SVD of the base weight (GPU, randomised).

    This is the SVD part of the paper's initialisation (Eq. 11-12: ``B = U_W sqrt(S_W)``,
    ``A = sqrt(S_W) V_W^T``). The paper's *task-specific* step, ``B = U U^T U_W sqrt(S_W)`` with
    ``U`` the top left singular vectors of ``W X_t`` on fine-tuning-data features (Eq. 8-10),
    is not implemented; ``task_init`` therefore means "SVD initialisation".
    """
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
        mod._salora_hook_handles = [  # type: ignore[attr-defined]
            sub.register_forward_hook(hook_fn) for sub in items
        ]

        mod._salora_installed = True  # type: ignore[attr-defined]
        installed += 1
        
    logger.info("salora: installed fixed safety module C in %d LoRA layers.", installed)
    return installed


# ---------------------------------------------------------------------------
# Residual re-parameterisation W' = W - C B A (paper Eq. 13)
# ---------------------------------------------------------------------------

def offset_base_weights(model: nn.Module) -> int:
    """Subtract the initial adapter's contribution from the base weight: ``W' = W - s * C B A``.

    After the task-specific initialisation ``B, A`` are non-zero, so the model with the
    safety module would no longer equal the pretrained one. Eq. 13 re-parameterises the layer
    so that it does. ``s`` is the PEFT scaling of the adapter (``lora_alpha / r``, or
    ``lora_alpha / sqrt(r)`` with rsLoRA), ``C`` the installed safety module (for
    ``strength != 1`` the blended one the hook applies, so the identity still holds).
    Idempotent; returns the number of layers offset.
    """
    done = 0
    for _name, mod in list(model.named_modules()):
        if not getattr(mod, "_salora_installed", False) or getattr(mod, "_salora_offset", False):
            continue
        C = getattr(mod, "lora_C", None)
        base = getattr(mod, "base_layer", None)
        weight = getattr(base, "weight", None)
        if C is None or weight is None or weight.dim() != 2 or not weight.is_floating_point():
            continue
        delta = torch.zeros_like(weight, dtype=torch.float32)
        for adapter in list(mod.lora_A.keys()):
            if adapter not in mod.lora_B:
                continue
            A = mod.lora_A[adapter].weight.detach().float()
            B = mod.lora_B[adapter].weight.detach().float()
            step = float(mod.scaling[adapter]) * (C.to(device=B.device, dtype=torch.float32) @ (B @ A))
            # Conv1D (GPT-2) stores the base weight as (in, out): PEFT's fan_in_fan_out.
            delta += step.T if getattr(mod, "fan_in_fan_out", False) else step
        with torch.no_grad():
            weight.sub_(delta.to(device=weight.device, dtype=weight.dtype))
        mod._salora_offset = True  # type: ignore[attr-defined]
        done += 1
    if done:
        logger.info("salora: offset the base weight (W' = W - C B A) in %d LoRA layers.", done)
    return done


# ---------------------------------------------------------------------------
# Folding the safety module into the adapter (paper Sec. 4.3)
# ---------------------------------------------------------------------------

def fold_safety_module(model: nn.Module) -> int:
    """Fold the fixed safety module into ``lora_B`` (``B <- C B``) and drop the forward hook.

    SaLoRA applies ``C`` in a forward hook on ``lora_B``, so ``merge_and_unload()`` (which adds
    ``scaling * B A``) and any adapter-only save would lose it. The paper (Sec. 4.3, Fig. 4)
    saves ``B' = C B`` instead. Call this once, after training and before merging or saving.
    Idempotent; returns the number of LoRA layers folded.
    """
    folded = 0
    for _name, mod in list(model.named_modules()):
        if not getattr(mod, "_salora_installed", False):
            continue
        C = getattr(mod, "lora_C", None)
        if C is None:
            continue
        for handle in getattr(mod, "_salora_hook_handles", []):
            handle.remove()
        mod._salora_hook_handles = []  # type: ignore[attr-defined]
        lora_B = getattr(mod, "lora_B", None)
        items = lora_B.values() if hasattr(lora_B, "values") else [lora_B]
        for sub in items:
            w = getattr(sub, "weight", None)
            if w is None:
                continue
            with torch.no_grad():
                w.copy_((C.to(device=w.device, dtype=torch.float32) @ w.detach().float()).to(w.dtype))
        del mod.lora_C
        mod._salora_installed = False  # type: ignore[attr-defined]
        folded += 1
    if folded:
        logger.info("salora: folded the safety module into lora_B for %d layers (B <- C B).", folded)
    return folded


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
            if apply_task_init:
                # Eq. 13: the model must still equal the pretrained one after the initialisation.
                offset_base_weights(model)
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
    "fold_safety_module",
    "offset_base_weights",
]