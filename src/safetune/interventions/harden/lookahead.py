"""LookAhead Tuning Trainer adapter.

Paper: "LookAhead Tuning: Safer Language Models via Partial Answer Previews",
Liu et al., arXiv:2503.19041. Repo: https://github.com/zjunlp/LookAheadTuning

LookAhead Tuning is a *data-driven* method: it modifies the training data so the
model previews a short prefix of the answer before generating it, which keeps the
initial-token distribution close to the base model and preserves safety alignment.

Two variants (paper Eq. 4 / Sec. 3):

* ``real``    -- new input  ``I' = I (+) connector (+) O[:m]`` : the first ``m``
                tokens of the *answer* are appended to the *instruction* side.
                The answer ``O`` is unchanged. Because the previewed tokens live
                on the instruction side, the standard CE loss is *not* computed
                on them, but the real initial answer tokens that follow are now
                conditioned on their own preview -- this is what lowers the loss
                on the critical initial answer tokens.
* ``virtual`` -- a fixed generic phrase ``P`` (default
                ``"Let's solve this problem. "``) is inserted on the instruction
                side ``I' = I (+) connector (+) P`` and *also* prepended to the
                answer ``O' = P (+) O``.

The standard cross-entropy SFT objective is unchanged; only the data is modified.

This module reimplements the answer-aware collator locally (``AnswerPreviewCollator``)
because the previewed prefix must be inserted *between the prompt and the answer*
and the previewed tokens must be the *real answer tokens* (or the phrase ``P``),
not a run of token-id-0.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

try:
    from transformers import Trainer, TrainingArguments
    from transformers.data.data_collator import default_data_collator
    _TRAINER_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    Trainer = object  # type: ignore[assignment,misc]
    TrainingArguments = object  # type: ignore[assignment,misc]
    default_data_collator = None  # type: ignore[assignment]
    _TRAINER_IMPORT_ERROR = _e

try:
    import torch
    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = _e

try:
    from safetune.core.data_compiler.lookahead import (
        LookAheadConfig as _CoreLookAheadConfig,
    )
    _LOOKAHEAD_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    _CoreLookAheadConfig = None  # type: ignore[assignment]
    _LOOKAHEAD_IMPORT_ERROR = _e


# Paper default virtual-answer phrase P (LookAhead Tuning-virtual).
DEFAULT_VIRTUAL_PREFIX = "Let's solve this problem. "


if _TRAINER_IMPORT_ERROR is None:
    from dataclasses import dataclass
    from typing import List as _List

    @dataclass
    class LookAheadConfig(TrainingArguments):  # type: ignore[misc]
        # "real": preview the first ``prefix_length`` tokens of the answer.
        # "virtual": preview the fixed phrase ``prefix_text`` / ``prefix_token_ids``.
        prefix_mode: str = "virtual"
        prefix_length: int = 6
        # Explicit token ids for the virtual phrase P (takes priority over text).
        prefix_token_ids: Optional[_List[int]] = None
        # Source text of the virtual phrase P (tokenized with the trainer tokenizer).
        prefix_text: str = DEFAULT_VIRTUAL_PREFIX
else:  # pragma: no cover
    LookAheadConfig = _CoreLookAheadConfig  # type: ignore[assignment]


def _answer_start(labels: List[int]) -> int:
    """Index of the first answer token (first label != -100).

    In SFT batches the prompt tokens carry label -100 and the answer tokens carry
    their real ids; the answer preview must be inserted at this boundary. If every
    label is supervised (no prompt mask) the answer starts at position 0.
    """
    for i, v in enumerate(labels):
        if int(v) != -100:
            return i
    return len(labels)


class AnswerPreviewCollator:
    """Insert a *real answer-prefix* (or the virtual phrase ``P``) preview between
    the prompt and the answer, as described in LookAhead Tuning (arXiv:2503.19041).

    ``real`` mode    -- inserts the first ``prefix_length`` answer tokens just
                        before the answer, labelled -100 (no loss): the model
                        previews the answer, the real answer tokens still train.
    ``virtual`` mode -- inserts the fixed phrase ``P`` (``prefix_token_ids``)
                        before the answer, labelled -100 on the instruction-side
                        copy; the answer ``O`` itself is left intact (the paper's
                        ``O' = P (+) O`` is realised by the preview sitting
                        immediately before, supervised-free, ahead of ``O``).
    """

    def __init__(
        self,
        base_collator: Callable[[List[Any]], Any],
        config: Optional[Any] = None,
    ) -> None:
        if _TORCH_IMPORT_ERROR is not None:  # pragma: no cover
            raise ImportError(
                "torch is required for AnswerPreviewCollator"
            ) from _TORCH_IMPORT_ERROR
        self.base_collator = base_collator
        self.config = config
        self.prefix_mode = str(getattr(config, "prefix_mode", "virtual") or "virtual")
        self.prefix_length = int(getattr(config, "prefix_length", 6) or 6)
        self.prefix_token_ids = getattr(config, "prefix_token_ids", None)

    def _virtual_prefix_ids(self) -> List[int]:
        if self.prefix_token_ids:
            return list(self.prefix_token_ids)
        # No phrase tokens supplied: fall back to repeating the answer's own first
        # token below (handled per-example). An empty list means "use real preview".
        return []

    def _preview_for_example(self, input_ids: List[int], labels: List[int]) -> List[int]:
        """Return the token ids previewed for one example."""
        start = _answer_start(labels)
        answer_ids = list(input_ids[start:])
        m = max(0, min(self.prefix_length, len(answer_ids)))
        if self.prefix_mode == "real":
            # Preview the actual first m answer tokens.
            return answer_ids[:m]
        # virtual mode
        virtual = self._virtual_prefix_ids()
        if virtual:
            return virtual
        # No explicit phrase tokens: preview the real answer prefix as a safe
        # fallback (still the answer, never token-id-0).
        return answer_ids[:m]

    def _insert_preview(self, example: Any) -> Any:
        if not isinstance(example, dict):
            return example
        if "input_ids" not in example or "labels" not in example:
            return example
        updated = dict(example)

        ids = example["input_ids"]
        lbl = example["labels"]
        was_tensor = torch is not None and isinstance(ids, torch.Tensor)
        ids_list = ids.tolist() if was_tensor else list(ids)
        lbl_list = lbl.tolist() if (torch is not None and isinstance(lbl, torch.Tensor)) else list(lbl)

        start = _answer_start(lbl_list)
        preview = self._preview_for_example(ids_list, lbl_list)
        plen = len(preview)
        if plen == 0:
            return updated

        # Insert the preview *between prompt and answer* (paper: I' = I (+) preview).
        new_ids = ids_list[:start] + list(preview) + ids_list[start:]
        # The previewed tokens are on the instruction side -> no loss (-100).
        new_lbl = lbl_list[:start] + [-100] * plen + lbl_list[start:]

        if "attention_mask" in updated:
            am = example["attention_mask"]
            am_list = am.tolist() if (torch is not None and isinstance(am, torch.Tensor)) else list(am)
            new_am = am_list[:start] + [1] * plen + am_list[start:]
        else:
            new_am = None

        if was_tensor:
            updated["input_ids"] = torch.tensor(new_ids, dtype=ids.dtype, device=ids.device)
            updated["labels"] = torch.tensor(
                new_lbl,
                dtype=(lbl.dtype if isinstance(lbl, torch.Tensor) else torch.long),
            )
            if new_am is not None:
                am = example["attention_mask"]
                updated["attention_mask"] = torch.tensor(
                    new_am,
                    dtype=(am.dtype if isinstance(am, torch.Tensor) else torch.long),
                )
        else:
            updated["input_ids"] = new_ids
            updated["labels"] = new_lbl
            if new_am is not None:
                updated["attention_mask"] = new_am
        return updated

    def __call__(self, features: List[Any]) -> Any:
        previewed = [self._insert_preview(f) for f in features]
        return self.base_collator(previewed)


class LookAheadTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """HF ``Trainer`` that applies LookAhead Tuning data augmentation.

    Public signature is unchanged: extra behaviour is driven entirely by optional
    fields on the ``args`` config (``prefix_mode``, ``prefix_length``,
    ``prefix_token_ids``, ``prefix_text``) with safe defaults, so passing a plain
    ``TrainingArguments`` still works.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for LookAheadTrainer"
            ) from _TRAINER_IMPORT_ERROR

        user_collator = kwargs.get("data_collator", None)
        tokenizer = kwargs.get("tokenizer", None) or kwargs.get("processing_class", None)
        args_obj = kwargs.get("args", None)
        if len(args) >= 2 and args_obj is None:
            args_obj = args[1]

        # Resolve LookAhead config from the args object (optional, with defaults).
        prefix_mode = str(getattr(args_obj, "prefix_mode", "virtual") or "virtual")
        prefix_length = int(getattr(args_obj, "prefix_length", 6) or 6)
        prefix_token_ids = getattr(args_obj, "prefix_token_ids", None)
        prefix_text = getattr(args_obj, "prefix_text", DEFAULT_VIRTUAL_PREFIX)

        # In virtual mode, tokenize the phrase P with the trainer's tokenizer if
        # explicit ids were not supplied.
        if (
            prefix_mode == "virtual"
            and not prefix_token_ids
            and tokenizer is not None
            and prefix_text
            and hasattr(tokenizer, "encode")
        ):
            try:
                prefix_token_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            except Exception:  # pragma: no cover - tokenizer quirks
                prefix_token_ids = None

        # Lightweight config carrier (avoid re-instantiating TrainingArguments).
        class _Cfg:  # noqa: D401
            pass

        cfg = _Cfg()
        cfg.prefix_mode = prefix_mode
        cfg.prefix_length = prefix_length
        cfg.prefix_token_ids = prefix_token_ids

        base_collator = user_collator if user_collator is not None else default_data_collator
        kwargs["data_collator"] = AnswerPreviewCollator(base_collator, config=cfg)

        super().__init__(*args, **kwargs)


__all__ = [
    "LookAheadConfig",
    "LookAheadTrainer",
    "AnswerPreviewCollator",
    "DEFAULT_VIRTUAL_PREFIX",
]
