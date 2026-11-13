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

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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


def _template_rejects_system(tokenizer: Any) -> bool:
    """True if the tokenizer's chat template raises on a ``system`` message (e.g. Gemma)."""
    if tokenizer is None or not getattr(tokenizer, "chat_template", None):
        return False
    from jinja2.exceptions import TemplateError

    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            tokenize=False,
        )
    except TemplateError:
        return True
    return False


def _fold_system(prompt: Any) -> Any:
    """``[system, user]`` -> ``[user]`` with the system text prepended to the user turn."""
    if isinstance(prompt, list) and len(prompt) >= 2 and prompt[0].get("role") == "system":
        first = dict(prompt[1])
        first["content"] = prompt[0]["content"] + "\n\n" + first["content"]
        return [first] + list(prompt[2:])
    return prompt


def prepare_cst_dataset(
    examples: List[Dict[str, str]],
    safe_system_prompt: Optional[str] = None,
    uncensored_system_prompt: Optional[str] = None,
    include_uncensored_pairs: bool = True,
    tokenizer: Any = None,
) -> List[Dict[str, Any]]:
    """Format ``examples`` into CST DPO pairs before training.

    Each example must have keys ``prompt``, ``safe_response``, ``unsafe_response``.

    TRL's ``DPOTrainer`` reads only ``prompt``, ``chosen`` and ``rejected``, and CST
    works only if the system prompt is part of the DPO prompt: the same user prompt
    under the safe and the uncensored system prompt has opposite preferences. Rows are
    therefore conversational (``prompt=[system, user]``, ``chosen``/``rejected`` =
    ``[assistant]``), which TRL renders with the model's chat template, as the
    authors' ``cst_train.py`` does. If ``tokenizer`` is given and has no chat template,
    the system prompt is folded into a plain-text prompt instead.
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
    rows = CSTFormatter(cfg).format_dataset(examples)
    plain = tokenizer is not None and not getattr(tokenizer, "chat_template", None)
    fold = _template_rejects_system(tokenizer)
    if fold:
        logger.warning(
            "prepare_cst_dataset: the chat template rejects the system role; the CST system "
            "prompt is folded into the first user turn."
        )
    out: List[Dict[str, Any]] = []
    for r in rows:
        if plain:
            out.append({"prompt": r["formatted_input"], "chosen": r["chosen"],
                        "rejected": r["rejected"], "cst_mode": r["cst_mode"]})
        else:
            out.append({
                "prompt": (_fold_system if fold else (lambda x: x))(
                    [{"role": "system", "content": r["system"]},
                     {"role": "user", "content": r["prompt"]}]),
                "chosen": [{"role": "assistant", "content": r["chosen"]}],
                "rejected": [{"role": "assistant", "content": r["rejected"]}],
                "cst_mode": r["cst_mode"],
            })
    return out


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

        # Name positional arguments so the datasets and the tokenizer are found either way.
        import inspect

        for name, value in zip(list(inspect.signature(DPOTrainer.__init__).parameters)[1:], args):
            if name in kwargs:
                raise TypeError(f"CSTTrainer.__init__() got multiple values for argument '{name}'")
            kwargs[name] = value

        if raw_examples is not None and kwargs.get("train_dataset") is None:
            # prepare_cst_dataset returns a plain list[dict]; DPOTrainer expects a
            # datasets.Dataset (it calls .map on train_dataset). Wrap it so passing
            # raw_examples works as documented instead of raising
            # "'list' object has no attribute 'map'".
            from datasets import Dataset
            kwargs["train_dataset"] = Dataset.from_list(
                prepare_cst_dataset(raw_examples, tokenizer=kwargs.get("processing_class")))

        # A dataset prepared without the tokenizer still carries the system role.
        if _template_rejects_system(kwargs.get("processing_class")):
            for key in ("train_dataset", "eval_dataset"):
                ds = kwargs.get(key)
                if ds is None:
                    continue
                if isinstance(ds, dict):
                    kwargs[key] = {k: _fold_dataset(v, key) for k, v in ds.items()}
                else:
                    kwargs[key] = _fold_dataset(ds, key)

        super().__init__(**kwargs)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        # DPOTrainer.log adds the DPO eval metrics under a fixed ``eval_`` prefix, so with a
        # dict ``eval_dataset`` every set wrote ``eval_rewards/...`` and the last one won
        # (huggingface/trl#8307). Take the set's prefix from its loss key instead, e.g.
        # ``eval_alpha_loss`` -> ``eval_alpha_rewards/...``. No-op on TRL versions without
        # ``_metrics``.
        bucket = (getattr(self, "_metrics", None) or {}).get("eval")
        loss_keys = [k for k in logs if k.startswith("eval_") and k.endswith("_loss")]
        if bucket and loss_keys:
            name = min(loss_keys, key=len)[len("eval_"):-len("loss")]   # "alpha_", or "" for one set
            if name:
                for key in list(bucket):
                    bucket[name + key] = bucket.pop(key)
        super().log(logs, start_time)


def _fold_dataset(ds: Any, name: str) -> Any:
    """Fold the system turn of a conversational ``prompt`` column (see :func:`_fold_system`)."""
    if not hasattr(ds, "map"):
        return ds
    # Peek with next(iter()) rather than len()/[0]: an IterableDataset has neither.
    first = (next(iter(ds), None) or {}).get("prompt")
    if isinstance(first, list) and first and first[0].get("role") == "system":
        logger.warning(
            "CSTTrainer: the chat template rejects the system role; the CST system "
            "prompt is folded into the first user turn of %s.", name,
        )
        return ds.map(lambda r: {"prompt": _fold_system(r["prompt"])})
    return ds
