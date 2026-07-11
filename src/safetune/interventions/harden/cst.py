"""
CST DPO Trainer adapter.

Thin wrapper around :class:`trl.DPOTrainer` that documents the CST
(Configurable Safety Tuning) data formatting step from
``safetune.core.data_compiler.cst``.

CST operates at the data level: the training set must be formatted with
:class:`CSTFormatter` into opposite-system-prompt DPO pairs *before*
being passed to this trainer. Use :func:`prepare_cst_dataset` for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from trl import DPOTrainer, DPOConfig
    _TRL_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    DPOTrainer = object  # type: ignore[assignment,misc]
    DPOConfig = object  # type: ignore[assignment,misc]
    _TRL_IMPORT_ERROR = _e

try:
    from safetune.core.data_compiler.cst import (
        CSTConfig as _CoreCSTConfig,
        CSTFormatter,
    )
    _CST_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    _CoreCSTConfig = None  # type: ignore[assignment]
    CSTFormatter = None  # type: ignore[assignment]
    _CST_IMPORT_ERROR = _e


if _TRL_IMPORT_ERROR is None:
    @dataclass
    class CSTConfig(DPOConfig):  # type: ignore[misc]
        """DPOConfig subclass for CST. CST is data-centric; no extra hyperparameters."""

        pass
else:  # pragma: no cover
    class CSTConfig(object):  # type: ignore[assignment]
        pass


def prepare_cst_dataset(
    examples: List[Dict[str, str]],
    safe_system_prompt: Optional[str] = None,
    uncensored_system_prompt: Optional[str] = None,
    include_uncensored_pairs: bool = True,
) -> List[Dict[str, Any]]:
    """Format ``examples`` into CST DPO pairs before training.

    Each example must have keys ``prompt``, ``safe_response``, ``unsafe_response``.
    """
    if _CST_IMPORT_ERROR is not None:
        raise ImportError(
            "safetune.core.data_compiler.cst is unavailable"
        ) from _CST_IMPORT_ERROR
    kwargs: Dict[str, Any] = {"include_uncensored_pairs": include_uncensored_pairs}
    if safe_system_prompt is not None:
        kwargs["safe_system_prompt"] = safe_system_prompt
    if uncensored_system_prompt is not None:
        kwargs["uncensored_system_prompt"] = uncensored_system_prompt
    cfg = _CoreCSTConfig(**kwargs)
    return CSTFormatter(cfg).format_dataset(examples)


class CSTTrainer(DPOTrainer if _TRL_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """DPOTrainer with optional on-init CST dataset formatting.

    If ``raw_examples`` is supplied, they are formatted via
    :func:`prepare_cst_dataset` and used as the training dataset.
    """

    def __init__(
        self,
        *args: Any,
        raw_examples: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> None:
        if _TRL_IMPORT_ERROR is not None:
            raise ImportError(
                "trl is required for CSTTrainer"
            ) from _TRL_IMPORT_ERROR
        if _CST_IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.data_compiler.cst is unavailable"
            ) from _CST_IMPORT_ERROR

        if raw_examples is not None and "train_dataset" not in kwargs:
            # prepare_cst_dataset returns a plain list[dict]; DPOTrainer expects a
            # datasets.Dataset (it calls .map on train_dataset). Wrap it so passing
            # raw_examples works as documented instead of raising
            # "'list' object has no attribute 'map'".
            from datasets import Dataset
            kwargs["train_dataset"] = Dataset.from_list(prepare_cst_dataset(raw_examples))

        super().__init__(*args, **kwargs)
