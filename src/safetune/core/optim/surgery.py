"""Surgery: attention-sink divergence regularizer for safe fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class SurgeryConfig:
    sink_lambda: float = 0.01


class SurgeryWrapper:
    def __init__(self, config: Optional[SurgeryConfig] = None) -> None:
        self.config = config or SurgeryConfig()
        self.ref_sinks: Optional[List[torch.Tensor]] = None

    def compute_sink_penalty(
        self, attentions: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        if attentions is None or len(attentions) == 0:
            return torch.tensor(0.0)

        per_layer_sinks: List[torch.Tensor] = []
        for attn in attentions:
            if attn is None:
                continue
            sink = attn[:, :, :, 0].mean(dim=(0, 2))
            per_layer_sinks.append(sink)

        if not per_layer_sinks:
            return torch.tensor(0.0)

        device = per_layer_sinks[0].device
        dtype = per_layer_sinks[0].dtype

        if self.ref_sinks is None:
            self.ref_sinks = [torch.zeros_like(s) for s in per_layer_sinks]

        total = torch.zeros((), device=device, dtype=dtype)
        for curr, ref in zip(per_layer_sinks, self.ref_sinks):
            ref_dev = ref.to(device=device, dtype=dtype)
            total = total + F.relu(curr - ref_dev).sum()

        return self.config.sink_lambda * total

    def set_reference(self, attentions: Tuple[torch.Tensor, ...]) -> None:
        refs: List[torch.Tensor] = []
        for attn in attentions:
            if attn is None:
                continue
            refs.append(attn[:, :, :, 0].mean(dim=(0, 2)).detach().clone())
        self.ref_sinks = refs
