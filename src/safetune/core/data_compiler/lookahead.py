"""LookAhead Tuning: prefix-prepending data collator to preserve initial-token distribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

try:
    import torch
    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = _e


@dataclass
class LookAheadConfig:
    prefix_mode: str = "virtual"
    prefix_length: int = 10
    prefix_token_ids: Optional[List[int]] = None


class LookAheadCollator:
    def __init__(
        self,
        base_collator: Callable[[List[Any]], Any],
        config: Optional[LookAheadConfig] = None,
    ) -> None:
        if _TORCH_IMPORT_ERROR is not None:
            raise ImportError(
                "torch is required for LookAheadCollator"
            ) from _TORCH_IMPORT_ERROR
        self.base_collator = base_collator
        self.config = config or LookAheadConfig()

    def _resolve_prefix_ids(self) -> List[int]:
        if self.config.prefix_token_ids:
            return list(self.config.prefix_token_ids)
        return [0] * int(self.config.prefix_length)

    def _prepend_example(self, example: Any, prefix_ids: List[int]) -> Any:
        if not isinstance(example, dict):
            return example
        updated = dict(example)
        plen = len(prefix_ids)
        if "input_ids" in updated:
            ids = updated["input_ids"]
            if isinstance(ids, list):
                updated["input_ids"] = list(prefix_ids) + list(ids)
            elif torch is not None and isinstance(ids, torch.Tensor):
                pref = torch.tensor(prefix_ids, dtype=ids.dtype, device=ids.device)
                updated["input_ids"] = torch.cat([pref, ids], dim=-1)
        if "attention_mask" in updated:
            am = updated["attention_mask"]
            if isinstance(am, list):
                updated["attention_mask"] = [1] * plen + list(am)
            elif torch is not None and isinstance(am, torch.Tensor):
                pref_am = torch.ones(plen, dtype=am.dtype, device=am.device)
                updated["attention_mask"] = torch.cat([pref_am, am], dim=-1)
        if "labels" in updated:
            lbl = updated["labels"]
            if isinstance(lbl, list):
                updated["labels"] = [-100] * plen + list(lbl)
            elif torch is not None and isinstance(lbl, torch.Tensor):
                pref_lbl = torch.full((plen,), -100, dtype=lbl.dtype, device=lbl.device)
                updated["labels"] = torch.cat([pref_lbl, lbl], dim=-1)
        return updated

    def __call__(self, features: List[Any]) -> Any:
        prefix_ids = self._resolve_prefix_ids()
        prepended = [self._prepend_example(f, prefix_ids) for f in features]
        return self.base_collator(prepended)


__all__ = ["LookAheadConfig", "LookAheadCollator"]
