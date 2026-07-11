"""Safe LoRA: project unsafe LoRA updates onto the safety-aligned subspace.

Faithful reimplementation of Hsu et al., NeurIPS 2024 (arXiv:2405.16833).
"""
from __future__ import annotations

import functools
import os
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

_F = TypeVar("_F", bound=Callable[..., nn.Module])

def assert_mutates(fn_name: str) -> Callable[[_F], _F]:
    def decorator(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(model: nn.Module, *args: Any, **kwargs: Any) -> nn.Module:
            before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            result = fn(model, *args, **kwargs)
            if before:
                mutated = any(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters() if n in before)
                if not mutated:
                    # A no-op is legitimate (e.g. every layer's drift already lies
                    # in the safety subspace). Follow the _invariant.assert_mutates
                    # contract: warn, don't break optional pipelines.
                    import logging
                    logging.getLogger(__name__).warning(
                        "%s: model was not mutated (no layer fell below the "
                        "projection threshold, or no base_state_dict was given).",
                        fn_name,
                    )
            return result
        return wrapper
    return decorator

def _load_state_dict(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return obj

def _to_2d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 2: return t
    return t.reshape(t.shape[0], -1)

def _compute_projection(a: torch.Tensor, b: torch.Tensor, device: torch.device) -> Optional[torch.Tensor]:
    if a.shape != b.shape: return None
    a_f = a.to(device=device, dtype=torch.float32)
    b_f = b.to(device=device, dtype=torch.float32)
    vec = _to_2d(a_f - b_f)
    
    # Safe-LoRA (Hsu et al., arXiv:2405.16833): C = V·Vᵀ / ‖V‖_F
    # (Frobenius norm, NOT squared). Use eps to guard div-by-zero.
    fro_norm = torch.sqrt(torch.sum(vec ** 2))
    if fro_norm <= 0: return None
    return torch.mm(vec, vec.t()) / (fro_norm + 1e-12)

def _find_sd_keys(sd: Dict[str, Any], module_key: str) -> Optional[str]:
    exact = module_key + ".weight"
    if exact in sd: return exact
    for k in sd.keys():
        if not k.endswith(".weight"): continue
        base_k = k[: -len(".weight")]
        if module_key.endswith(base_k) or base_k.endswith(module_key): return k
    return None

def _module_key_for_lora(lora_name: str) -> str:
    key = lora_name
    for marker in (".lora_A", ".lora_B", ".lora_embedding_A", ".lora_embedding_B"):
        if marker in key: 
            key = key.split(marker)[0]
            break
    if key.startswith("base_model.model."): 
        key = key[len("base_model.model."):]
    return key

def _collect_lora_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    layers: List[Tuple[str, nn.Module]] = []
    for mod_name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            la, lb = getattr(module, "lora_A"), getattr(module, "lora_B")
            if hasattr(la, "keys") and hasattr(lb, "keys"):
                layers.append((mod_name, module))
    return layers

def _project_peft_model(model, aligned_sd, base_sd, target_modules, threshold, select_layers_type, num_proj_layers):
    lora_layers = _collect_lora_layers(model)
    records = []
    for mod_name, module in lora_layers:
        module_key = _module_key_for_lora(mod_name)
        if not any(mod in module_key for mod in target_modules): continue
        ak = _find_sd_keys(aligned_sd, module_key)
        bk = _find_sd_keys(base_sd, module_key)
        if not ak or not bk: continue
        adapters = [adp for adp in module.lora_A.keys() if adp in module.lora_B and module.lora_A[adp].weight.dim() == 2 and module.lora_B[adp].weight.dim() == 2]
        if not adapters: continue
        target_device = module.lora_A[adapters[0]].weight.device
        proj = _compute_projection(aligned_sd[ak], base_sd[bk], target_device)
        if proj is None: continue
        if proj.shape[0] != module.lora_B[adapters[0]].weight.shape[0]: 
            del proj
            continue
        
        for adp in adapters:
            a_w, b_w = module.lora_A[adp].weight, module.lora_B[adp].weight
            with torch.no_grad():
                delta = torch.mm(b_w.detach().float(), a_w.detach().float())
                proj_delta = torch.mm(proj, delta)
                cos = F.cosine_similarity(proj_delta.reshape(1, -1), delta.reshape(1, -1)).item()
            records.append({"module": mod_name, "adapter": adp, "lora_B": module.lora_B[adp], "cos": float(cos), "ak": ak, "bk": bk, "target_device": target_device})
        del proj

    effective_threshold = float(threshold)
    if select_layers_type == "number" and records:
        cosines = sorted(r["cos"] for r in records)
        n = max(0, min(int(num_proj_layers), len(cosines)))
        effective_threshold = float("-inf") if n == 0 else cosines[n - 1]
    
    projected = 0
    for rec in records:
        if rec["cos"] <= effective_threshold:
            proj = _compute_projection(aligned_sd[rec["ak"]], base_sd[rec["bk"]], rec["target_device"])
            if proj is None: continue
            with torch.no_grad():
                b_w = rec["lora_B"].weight
                b_w.copy_(torch.mm(proj.to(dtype=b_w.dtype), b_w))
            projected += 1
            del proj
            
    # FIX: Empty cache only once at the end of the entire loop to prevent GPU stalling
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    return {"mode": "peft", "lora_layers_seen": len(records), "layers_projected": projected,
            "select_layers_type": select_layers_type, "effective_threshold": effective_threshold}

def _project_state_dict_model(model, aligned_sd, base_sd, target_modules, threshold, select_layers_type, num_proj_layers):
    named = dict(model.named_parameters())
    records = []
    for pname, param in named.items():
        if not pname.endswith(".weight"): continue
        if not any(mod in pname for mod in target_modules): continue
        mod_key = pname[: -len(".weight")]
        ak = _find_sd_keys(aligned_sd, mod_key)
        bk = _find_sd_keys(base_sd, mod_key)
        if not ak or not bk: continue
        target_device = param.device
        proj = _compute_projection(aligned_sd[ak], base_sd[bk], target_device)
        if proj is None: continue
        with torch.no_grad():
            base_2d = _to_2d(base_sd[bk].to(target_device).float())
            cur_2d = _to_2d(param.detach().float())
            if base_2d.shape != cur_2d.shape or proj.shape[0] != cur_2d.shape[0]: 
                del proj
                continue
            delta = cur_2d - base_2d
            proj_delta = torch.mm(proj, delta)
            cos = F.cosine_similarity(proj_delta.reshape(1, -1), delta.reshape(1, -1)).item()
            records.append({"name": pname, "param": param, "cos": float(cos), "ak": ak, "bk": bk, "target_device": target_device})
        del proj; del base_2d; del cur_2d; del delta; del proj_delta

    effective_threshold = float(threshold)
    if select_layers_type == "number" and records:
        cosines = sorted(r["cos"] for r in records)
        n = max(0, min(int(num_proj_layers), len(cosines)))
        effective_threshold = float("-inf") if n == 0 else cosines[n - 1]
    
    projected = 0
    for rec in records:
        if rec["cos"] <= effective_threshold:
            param = rec["param"]
            proj = _compute_projection(aligned_sd[rec["ak"]], base_sd[rec["bk"]], rec["target_device"])
            if proj is None: continue
            with torch.no_grad():
                base_2d = _to_2d(base_sd[rec["bk"]].to(rec["target_device"]).float())
                cur_2d = _to_2d(param.detach().float())
                delta = cur_2d - base_2d
                proj_delta = torch.mm(proj, delta)
                new_2d = base_2d + proj_delta
                param.data.copy_(new_2d.reshape(param.shape).to(param.dtype))
            projected += 1
            del proj; del base_2d; del cur_2d; del delta; del proj_delta
            
    # FIX: Empty cache only once at the end
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    return {"mode": "state_dict", "lora_layers_seen": len(records), "layers_projected": projected,
            "select_layers_type": select_layers_type, "effective_threshold": effective_threshold}

@assert_mutates("apply_safe_lora")
def apply_safe_lora(
    model: nn.Module,
    aligned_state_dict_path: Optional[str] = None,
    aligned_state_dict: Optional[Dict[str, Any]] = None,
    base_state_dict: Optional[Dict[str, Any]] = None,
    aligned_adapter_path: Optional[str] = None,
    base_adapter_path: Optional[str] = None,
    alpha: float = 0.5,
    max_delta_norm: Optional[float] = None,
    *,
    base_state_dict_path: Optional[str] = None,
    threshold: Optional[float] = None,
    select_layers_type: str = "threshold",
    num_proj_layers: int = 10,
    target_modules: Optional[List[str]] = None,
    **extra: Any,
) -> nn.Module:
    if threshold is None: 
        threshold = float(alpha) if alpha is not None else 0.5
    threshold = float(threshold)
    
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    is_peft = bool(_collect_lora_layers(model))

    # 1. FIX: Load base_sd first so we can use it to reconstruct the aligned adapter
    base_sd = None
    if base_state_dict is not None: 
        base_sd = dict(base_state_dict)
    elif base_state_dict_path is not None: 
        base_sd = _load_state_dict(base_state_dict_path)
    
    if base_sd is None:
        if is_peft:
            base_sd = {k: v.detach().clone().cpu() for k, v in model.state_dict().items() if "lora_" not in k}
        else:
            base_sd = {n: p.detach().clone().cpu() for n, p in model.named_parameters()}

    # 2. Load aligned_sd safely
    aligned_sd = None
    if aligned_state_dict is not None: 
        aligned_sd = dict(aligned_state_dict)
    elif aligned_state_dict_path is not None: 
        aligned_sd = _load_state_dict(aligned_state_dict_path)
    elif aligned_adapter_path is not None:
        # FIX: Avoid calling PeftModel.from_pretrained to prevent overwriting the active model
        try:
            from safetensors.torch import load_file
        except ImportError:
            load_file = None

        st_path = os.path.join(aligned_adapter_path, "adapter_model.safetensors")
        bin_path = os.path.join(aligned_adapter_path, "adapter_model.bin")
        if os.path.exists(st_path) and load_file is not None:
            adapter_sd = load_file(st_path)
        else:
            adapter_sd = torch.load(bin_path, map_location="cpu", weights_only=True)
            
        config_path = os.path.join(aligned_adapter_path, "adapter_config.json")
        with open(config_path, "r") as f:
            peft_config = json.load(f)
        
        # Lora scaling factor
        scaling = peft_config["lora_alpha"] / peft_config["r"]
        
        # Clone base_sd to build aligned full weights safely
        aligned_sd = {k: v.clone() for k, v in base_sd.items()}
        
        for key in adapter_sd.keys():
            if "lora_B" in key:
                a_key = key.replace("lora_B", "lora_A")
                if a_key in adapter_sd:
                    b_weight = adapter_sd[key].float()
                    a_weight = adapter_sd[a_key].float()
                    
                    base_key = _module_key_for_lora(key) + ".weight"
                    
                    if base_key in aligned_sd:
                        delta = torch.mm(b_weight, a_weight) * scaling
                        aligned_sd[base_key] = (aligned_sd[base_key].float() + delta).to(aligned_sd[base_key].dtype)

    if aligned_sd is None:
        raise ValueError("apply_safe_lora requires aligned model weights.")

    # 3. Route to proper projection handler
    if is_peft:
        _project_peft_model(model, aligned_sd, base_sd, target_modules=target_modules,
                            threshold=threshold, select_layers_type=select_layers_type, num_proj_layers=num_proj_layers)
        return model

    _project_state_dict_model(model, aligned_sd=aligned_sd, base_sd=base_sd, target_modules=target_modules,
                              threshold=threshold, select_layers_type=select_layers_type, num_proj_layers=num_proj_layers)
    return model

__all__ = ["apply_safe_lora"]