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
                answer ``O' = P (+) O``, with the loss on ``P`` and ``O``.

**Differences from the paper.** The runner receives tokenised rows, so the raw
instruction text is not available. The preview is therefore inserted into the token
stream between the prompt (including the chat template's assistant header) and the
answer, not into the user turn, and the connector ``" The answer begins with: "`` is not
added. Real mode inserts ``O[:m]`` (``m`` is clamped to the answer length) with no loss
on the copy; virtual mode inserts ``P`` once as an unsupervised preview and once as the
supervised start of ``O' = P (+) O``.

The standard cross-entropy SFT objective is unchanged; only the data is modified.

This module implements the collator locally (``AnswerPreviewCollator``): the previewed
prefix is inserted between the prompt and the answer, and the previewed tokens are the
real answer tokens (or the phrase ``P``), not a run of token-id-0.
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
    the prompt and the answer (see the module docstring for how this differs from
    LookAhead Tuning, arXiv:2503.19041).

    ``real`` mode    -- inserts the first ``min(prefix_length, len(answer))`` answer
                        tokens just before the answer, labelled -100 (no loss): the model
                        previews the answer, the real answer tokens still train.
    ``virtual`` mode -- inserts the fixed phrase ``P`` (``prefix_token_ids``) twice
                        before the answer: an unsupervised preview copy (-100) and a
                        supervised copy, so the loss covers ``O' = P (+) O``.
    """

    MODES = ("real", "virtual")

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
        if self.prefix_mode not in self.MODES:
            raise ValueError(
                f"prefix_mode must be one of {self.MODES}, got {self.prefix_mode!r}"
            )
        length = getattr(config, "prefix_length", 6)
        self.prefix_length = 6 if length is None else int(length)
        if self.prefix_length < 0:
            raise ValueError(f"prefix_length must be >= 0, got {self.prefix_length}")
        self.prefix_token_ids = getattr(config, "prefix_token_ids", None)
        # ponytail: id 0 without a tokenizer; the padded positions have mask 0 and label -100.
        self.pad_token_id = getattr(config, "pad_token_id", None) or 0

    def _preview_for_example(self, answer_ids: List[int]) -> List[int]:
        """Return the token ids previewed for one example (``answer_ids`` = supervised tokens)."""
        if self.prefix_mode == "real":
            # The first m real answer tokens; m is clamped to the answer length.
            return answer_ids[: self.prefix_length]
        if not self.prefix_token_ids:
            raise ValueError(
                "prefix_mode='virtual' needs the phrase P as token ids: pass "
                "prefix_token_ids, or a tokenizer (processing_class) so prefix_text can be encoded."
            )
        return list(self.prefix_token_ids)

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
        # Only supervised positions are answer tokens (padding carries label -100).
        answer_ids = [t for t, l in zip(ids_list[start:], lbl_list[start:]) if int(l) != -100]
        preview = self._preview_for_example(answer_ids)
        plen = len(preview)
        if plen == 0:
            return updated

        # Insert between prompt and answer. The first copy is the instruction-side preview
        # (no loss); in virtual mode a second, supervised copy starts O' = P (+) O.
        if self.prefix_mode == "virtual":
            ins_ids = list(preview) + list(preview)
            ins_lbl = [-100] * plen + list(preview)
        else:
            ins_ids = list(preview)
            ins_lbl = [-100] * plen
        n_ins = len(ins_ids)
        new_ids = ids_list[:start] + ins_ids + ids_list[start:]
        new_lbl = lbl_list[:start] + ins_lbl + lbl_list[start:]

        if "attention_mask" in updated:
            am = example["attention_mask"]
            am_list = am.tolist() if (torch is not None and isinstance(am, torch.Tensor)) else list(am)
            new_am = am_list[:start] + [1] * n_ins + am_list[start:]
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
        return self.base_collator(self._pad_to_longest(previewed))

    def _pad_to_longest(self, rows: List[Any]) -> List[Any]:
        """Right-pad rows to the longest one: a clamped preview makes their lengths differ."""
        fill = {"input_ids": self.pad_token_id, "labels": -100, "attention_mask": 0}
        longest = max((len(r["input_ids"]) for r in rows if isinstance(r, dict) and "input_ids" in r), default=0)
        padded = []
        for row in rows:
            if isinstance(row, dict):
                row = dict(row)
                for key, value in fill.items():
                    n = longest - len(row[key]) if key in row else 0
                    if n > 0 and torch is not None and isinstance(row[key], torch.Tensor):
                        row[key] = torch.cat([row[key], row[key].new_full((n,), value)])
                    elif n > 0:
                        row[key] = list(row[key]) + [value] * n
            padded.append(row)
        return padded


class LookAheadHFTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """HF ``Trainer`` that applies LookAhead Tuning data augmentation.

    Public signature is unchanged: extra behaviour is driven entirely by optional
    fields on the ``args`` config (``prefix_mode``, ``prefix_length``,
    ``prefix_token_ids``, ``prefix_text``) with safe defaults, so passing a plain
    ``TrainingArguments`` still works.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for LookAheadHFTrainer"
            ) from _TRAINER_IMPORT_ERROR

        user_collator = kwargs.get("data_collator", None)
        tokenizer = kwargs.get("tokenizer", None) or kwargs.get("processing_class", None)
        args_obj = kwargs.get("args", None)
        if len(args) >= 2 and args_obj is None:
            args_obj = args[1]

        # Resolve LookAhead config from the args object (optional, with defaults).
        prefix_mode = str(getattr(args_obj, "prefix_mode", "virtual") or "virtual")
        _len = getattr(args_obj, "prefix_length", 6)
        prefix_length = 6 if _len is None else int(_len)
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
        cfg.pad_token_id = getattr(tokenizer, "pad_token_id", None)

        base_collator = user_collator if user_collator is not None else default_data_collator
        kwargs["data_collator"] = AnswerPreviewCollator(base_collator, config=cfg)

        super().__init__(*args, **kwargs)


__all__ = [
    "LookAheadConfig",
    "LookAheadHFTrainer",
    "AnswerPreviewCollator",
    "DEFAULT_VIRTUAL_PREFIX",
]


from ._deprecated import renamed as _renamed

__getattr__ = _renamed(__name__, LookAheadTrainer="LookAheadHFTrainer")  # old name, remove in 0.3
