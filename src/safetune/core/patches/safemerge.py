"""SafeMERGE: layer-wise safety-aware merging via task/safety cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class SafeMergeConfig:
    threshold: float = 0.5
    alpha: float = 0.5


class SafeMergeWrapper:
    def __init__(
        self,
        base_state_dict: Dict[str, Any],
        aligned_state_dict: Dict[str, Any],
        config: Optional[SafeMergeConfig] = None,
    ) -> None:
        self.config = config or SafeMergeConfig()
        self.base_sd = base_state_dict
        self.aligned_sd = aligned_state_dict

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
        a_flat = a.reshape(-1).float()
        b_flat = b.reshape(-1).float()
        if a_flat.numel() == 0 or b_flat.numel() == 0:
            return None
        if torch.norm(a_flat) < 1e-12 or torch.norm(b_flat) < 1e-12:
            return None
        return float(F.cosine_similarity(a_flat, b_flat, dim=0).item())

    def apply(
        self,
        finetuned_state_dict: Dict[str, Any],
        threshold: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        thr = threshold if threshold is not None else self.config.threshold
        a = alpha if alpha is not None else self.config.alpha
        result: Dict[str, Any] = {}
        for key, w_ft in finetuned_state_dict.items():
            if key not in self.base_sd or key not in self.aligned_sd:
                result[key] = w_ft
                continue
            w_base = self.base_sd[key]
            w_aligned = self.aligned_sd[key]
            if w_ft.shape != w_base.shape or w_ft.shape != w_aligned.shape:
                result[key] = w_ft
                continue
            task_vec = w_ft.float() - w_base.float()
            safety_vec = w_aligned.float() - w_base.float()
            sim = self._cosine(task_vec, safety_vec)
            if sim is None:
                result[key] = w_ft
                continue
            if sim < thr:
                merged = (1.0 - a) * w_ft.float() + a * w_aligned.float()
                result[key] = merged.to(w_ft.dtype)
            else:
                result[key] = w_ft
        return result
