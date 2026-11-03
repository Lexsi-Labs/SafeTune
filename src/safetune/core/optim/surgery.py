"""Surgery: attention-sink divergence regularizer for safe fine-tuning.

Liu et al., arXiv:2602.05228, Eq. 2-4. The sink value of head ``h`` on data ``X`` is the mean
attention weight that query tokens ``i != k`` give to the sink token ``k``::

    alpha_h(X) = 1/|N_k| * sum_{i in N_k} A[h, k, i](X)          (Eq. 2, k = sink token = index 0)

and the regulariser is ``lambda * (1/|H|) * sum_h ReLU(alpha_h(X_m) - alpha_h(X_r))`` (Eq. 4), the mean
over **all** heads of all layers. Query row 0 (the sink itself) and padded query rows are excluded,
as in the official ``Lslland/Surgery`` ``trainer.py::_calculate_sink_mean``
(``head_attn[:, 1:, SINK_TOKEN_INDEX].mean()``). With an attention mask the sink is each row's first
attended token, so left-padded rows (where key 0 is a pad token with weight 0) are measured too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class SurgeryConfig:
    # Paper Table 8 (lambda = 300) and the official script (alpha=300), with the mean over heads.
    sink_lambda: float = 300.0


def sink_values(
    attention: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sink_index: int = 0,
) -> torch.Tensor:
    """Per-head sink value ``alpha_h`` of one layer.

    Args:
        attention: ``(batch, heads, T, T)`` attention weights (query, key).
        attention_mask: optional ``(batch, T)`` mask (1 = real token); padded query rows are
            excluded from the mean, and positions count from each row's first real token.
        sink_index: key position of the sink token (``args.sink_token_position`` upstream).

    Returns:
        ``(heads,)`` mean over the valid query rows after the sink row (and the batch) of the
        attention to the sink token.
    """
    if attention_mask is None:
        return attention[:, :, 1:, sink_index].mean(dim=(0, 2))
    mask = attention_mask.to(device=attention.device)
    first = mask.long().argmax(dim=1)                          # (B,) 0 for right padding
    B, H, T, _ = attention.shape
    key = (first + sink_index).view(B, 1, 1, 1).expand(B, H, T, 1)
    col = attention.gather(3, key).squeeze(3)                  # (B, H, T)
    m = mask.to(col.dtype).clone()
    m[torch.arange(B, device=m.device), first] = 0             # the sink's own query row
    m = m[:, None, :]                                          # (B, 1, T)
    return (col * m).sum(dim=(0, 2)) / m.sum().clamp_min(1.0)


class SurgeryWrapper:
    def __init__(self, config: Optional[SurgeryConfig] = None) -> None:
        self.config = config or SurgeryConfig()
        self.ref_sinks: Optional[List[torch.Tensor]] = None

    def compute_sink_penalty(
        self,
        attentions: Tuple[torch.Tensor, ...],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``lambda * mean_h ReLU(alpha_h(X_m) - alpha_h(X_r))`` over all heads of all layers."""
        if attentions is None or len(attentions) == 0:
            return torch.tensor(0.0)

        per_layer_sinks: List[torch.Tensor] = []
        for attn in attentions:
            if attn is None:
                continue
            per_layer_sinks.append(sink_values(attn, attention_mask))

        if not per_layer_sinks:
            return torch.tensor(0.0)

        device = per_layer_sinks[0].device
        dtype = per_layer_sinks[0].dtype

        if self.ref_sinks is None:
            self.ref_sinks = [torch.zeros_like(s) for s in per_layer_sinks]

        total = torch.zeros((), device=device, dtype=dtype)
        n_heads = 0
        for curr, ref in zip(per_layer_sinks, self.ref_sinks):
            ref_dev = ref.to(device=device, dtype=dtype)
            total = total + F.relu(curr - ref_dev).sum()
            n_heads += curr.numel()

        return self.config.sink_lambda * total / max(n_heads, 1)

    def set_reference(
        self,
        attentions: Tuple[torch.Tensor, ...],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        refs: List[torch.Tensor] = []
        for attn in attentions:
            if attn is None:
                continue
            refs.append(sink_values(attn, attention_mask).detach().clone())
        self.ref_sinks = refs
