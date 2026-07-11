"""Antibody — VESTIGIAL gradient-attenuation wrapper (NOT the paper's method).

⚠️ This cosine-similarity gradient scaler (``scale = 1 − cos(g, g_ref)``) is NOT
the Antibody algorithm and is NOT on any execution path: ``AntibodyWrapper`` is
constructed by ``harden/antibody.py`` but its ``attenuate_gradients`` /
``capture_harmful_reference`` are never called. The FAITHFUL Antibody
(arXiv:2603.00498, ICLR 2026 — SAM-flatness on harmful samples + adaptive-λ
sharpness + perturbed-model refusal loss, Eq.11) lives in
``safetune.harden.antibody.AntibodyTrainer``. Treat the code in this module as
dead/experimental scaffolding, not Antibody.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

try:
    from transformers import TrainingArguments
    _TA_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    TrainingArguments = object  # type: ignore[assignment,misc]
    _TA_IMPORT_ERROR = _e


if _TA_IMPORT_ERROR is None:
    @dataclass
    class AntibodyConfig(TrainingArguments):  # type: ignore[misc]
        sim_threshold: float = 0.0
        sam_rho: float = 0.05
else:  # pragma: no cover
    class AntibodyConfig(object):  # type: ignore[assignment]
        pass


class AntibodyWrapper:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.ref_grad: Optional[torch.Tensor] = None

    @staticmethod
    def _flat_grads(model: torch.nn.Module) -> Optional[torch.Tensor]:
        parts = []
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                parts.append(p.grad.detach().reshape(-1))
        if not parts:
            return None
        return torch.cat(parts)

    def capture_harmful_reference(
        self,
        model: torch.nn.Module,
        harmful_batch: Any,
        loss_fn: Optional[Callable[[torch.nn.Module, Any], torch.Tensor]] = None,
    ) -> None:
        model.zero_grad()
        if loss_fn is not None:
            loss = loss_fn(model, harmful_batch)
        else:
            if isinstance(harmful_batch, dict):
                out = model(**harmful_batch)
            else:
                out = model(harmful_batch)
            loss = out.loss if hasattr(out, "loss") else out
        loss.backward()
        flat = self._flat_grads(model)
        if flat is not None:
            self.ref_grad = flat.clone()
        model.zero_grad()

    def attenuate_gradients(self, model: torch.nn.Module, sim_threshold: float = 0.0) -> float:
        if self.ref_grad is None:
            return 0.0
        cur = self._flat_grads(model)
        if cur is None:
            return 0.0
        ref = self.ref_grad.to(cur.device)
        denom = (cur.norm() * ref.norm()).clamp_min(1e-12)
        sim = float(torch.dot(cur, ref) / denom)
        if sim > sim_threshold:
            scale = 1.0 - max(0.0, sim)
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.mul_(scale)
        return sim
