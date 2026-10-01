"""
DeRTa (Decoupled Refusal Training) trainer adapter.

Faithful implementation of the DeRTa objective from:

    "Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via
    Decoupled Refusal Training"
    Yuan, Jiao, Wang, Huang, Xu, Liang, He, Tu. ACL 2025. arXiv:2407.09121.
    Reference code: https://github.com/RobustNLP/DeRTa
      (``run_files/run_clm_lora_derta_llama.py`` -> ``MyLlamaForCausalLM``;
       ``data/train/generate_training_data.py`` -> data construction.)

DeRTa is a *supervised fine-tuning* (MLE) method, not a preference-optimisation
method.  It has two decoupled components:

  (1) **MLE with Harmful Response Prefix.** A variable-length segment of a
      harmful response is prepended to the safe refusal, so the model learns
      the standard next-token objective on a sequence that *starts* harmful and
      then refuses.  This teaches refusal at any position.  This component is a
      data-level augmentation and is produced by
      :class:`safetune.core.data_compiler.derta.DeRTaFormatter` (see
      :func:`prepare_derta_dataset`).

  (2) **Reinforced Transition Optimization (RTO).** This is a *per-token
      transition objective*, not extra prefix rows.  In the authors' code
      (``MyLlamaForCausalLM.forward``) every example carries, besides the
      ordinary ``labels``, a binary ``safe`` flag.  For examples flagged
      ``safe=true`` a second label stream ``safe_labels`` is built by
      replacing **every** non-ignored response token with the single
      transition-to-refusal token id (``19701`` for LLaMA-3, which is
      "Sorry").  A cross-entropy loss is then taken against
      ``safe_labels``.  This trains the model, at *every* position of the
      harmful continuation, to predict the transition-to-refusal token --
      reinforcing the harmful->safe transition continuously.

      The authors' relevant snippet::

          for bs, sl, label in zip(binary_safe, safe_labels, labels):
              if bs == 0:
                  continue            # standard MLE example
              else:
                  sl[label != -100] = 19701   # llama3 refusal token
          ...
          loss = CrossEntropyLoss()(shift_logits, shift_labels)

The earlier version of this adapter was a thin :class:`trl.DPOTrainer`
subclass that only applied component (1).  DeRTa emits SFT-style
``{prompt, response}`` rows (no ``chosen``/``rejected``), so a DPO base class
is a mismatch; and the per-token RTO objective -- the defining mechanism of
the paper -- was absent.  This rewrite:

  * bases :class:`DeRTaHFTrainer` on :class:`trl.SFTTrainer` (the correct
    SFT base, matching the authors' causal-LM training script);
  * implements RTO faithfully inside :meth:`DeRTaHFTrainer.compute_loss` as a
    real per-token transition cross-entropy against the refusal token id,
    mixed with the standard MLE loss.

The public construction signature in ``harden/__init__.py`` is unchanged --
all new behaviour is exposed through optional keyword arguments / config
fields that default to the paper's settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from trl import SFTTrainer, SFTConfig
    _TRL_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    SFTTrainer = object  # type: ignore[assignment,misc]
    SFTConfig = object  # type: ignore[assignment,misc]
    _TRL_IMPORT_ERROR = _e

try:
    from safetune.core.data_compiler.derta import (
        DeRTaConfig as _CoreDeRTaConfig,
        DeRTaFormatter,
        RTO_REFUSAL_TEXT,
        refusal_token_id,
    )
    _DERTA_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    _CoreDeRTaConfig = None  # type: ignore[assignment]
    DeRTaFormatter = None  # type: ignore[assignment]
    RTO_REFUSAL_TEXT, refusal_token_id = "Sorry", None  # type: ignore[assignment]
    _DERTA_IMPORT_ERROR = _e


# The authors hardcode 19701, which is "Sorry" in the Llama-3 tokenizer. The
# default ``rto_refusal_token_id=None`` derives the id from ``rto_refusal_text``
# with the trainer's tokenizer (19701 on Llama-3); pass
# ``rto_refusal_token_id=19701`` to force the old id on any tokenizer.


if _TRL_IMPORT_ERROR is None:
    @dataclass
    class DeRTaConfig(SFTConfig):  # type: ignore[misc]
        """:class:`trl.SFTConfig` subclass exposing the DeRTa / RTO hyper-parameters.

        Every field has a default, so the public construction signature is
        unchanged.  Defaults reproduce the paper's setting (RTO enabled, equal
        weighting of the MLE and RTO terms).
        """

        # Enable the Reinforced Transition Optimization term.  When False the
        # trainer is a plain SFT trainer over the (already prefix-augmented) data.
        enable_rto: bool = True
        # RTO reads the custom `safe` column at loss time; HF's default (True)
        # strips it (it isn't a forward() arg), silently disabling RTO for
        # direct-API users. The `_signature_columns` guard below is unreliable
        # (the attribute is None until lazily populated), so pin it here.
        remove_unused_columns: bool = False
        # Mixing weight of the RTO transition loss added on top of the standard
        # MLE loss.  The authors weight the two equally (1.0).
        rto_weight: float = 1.0
        # Token id the model must predict at every harmful-continuation position
        # (the transition-to-refusal token).  ``None`` -> resolve from the
        # tokenizer at run time using ``rto_refusal_text``.
        rto_refusal_token_id: Optional[int] = None
        # Text whose first token id is used as the transition token when
        # ``rto_refusal_token_id`` is None ("Sorry" -> 19701 on Llama-3).
        rto_refusal_text: str = RTO_REFUSAL_TEXT
        # True: the old objective, MLE over every response token (harmful
        # prefix included) plus rto_weight x a separate RTO loss on the
        # prefix + refusal rows. None: the authors' single cross-entropy (MLE
        # rows: safe response only; RTO rows: every token -> refusal token),
        # or the old one with safetune.configure(legacy_derta=True).
        legacy_derta: Optional[bool] = None
        # The legacy RTO term reads the logits. trl >= 1's default
        # "chunked_nll" returns none, which silently dropped that term; "nll"
        # (trl 0.x's default, same loss) keeps them.
        loss_type: str = "nll"

        def __post_init__(self):
            # TRL turns bf16=None into True whatever the hardware, so a bare
            # DeRTaConfig() raises on CPU. None follows the runtime dtype
            # instead (safetune.configure(device=..., dtype=...)); pass bf16
            # explicitly to override.
            if self.bf16 is None and not self.fp16:
                import torch
                from safetune.config import resolve_device, resolve_dtype
                device = resolve_device()
                self.bf16 = (resolve_dtype(device=device) == torch.bfloat16
                             and resolve_dtype("auto", device) == torch.bfloat16)
            super().__post_init__()
else:  # pragma: no cover
    class DeRTaConfig(object):  # type: ignore[assignment]
        pass


def prepare_derta_dataset(
    examples: List[Dict[str, str]],
    num_prefix_variants: int = 5,
    max_prefix_ratio: float = 0.8,
    enable_rto: bool = True,
    seed: int = 42,
    legacy: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Apply DeRTa data augmentation (component 1, MLE-with-harmful-prefix)
    to ``examples`` before training.

    Each example must have keys ``prompt``, ``harmful_response``,
    ``safe_response``.

    The returned rows carry a ``safe`` flag (``1`` for the RTO rows, ``0``
    otherwise) which :class:`DeRTaHFTrainer` reads to decide which rows get the
    refusal-token targets.  This mirrors the ``"safe"`` field in the authors'
    ``generate_training_data.py`` (``"true"`` only for ``Sorry_data``).
    ``legacy`` (default: ``safetune.configure(legacy_derta=...)``, False) gives
    the old rows, with ``safe=1`` on the prefix-augmented rows too.
    """
    if _DERTA_IMPORT_ERROR is not None:
        raise ImportError(
            "safetune.core.data_compiler.derta is unavailable"
        ) from _DERTA_IMPORT_ERROR
    if legacy is None:
        from safetune.config import get_config
        legacy = get_config().legacy_derta
    cfg = _CoreDeRTaConfig(
        num_prefix_variants=num_prefix_variants,
        max_prefix_ratio=max_prefix_ratio,
        enable_rto=enable_rto,
        seed=seed,
        legacy=legacy,
    )
    rows = DeRTaFormatter(cfg).augment_dataset(examples)

    # Tag each row with the binary ``safe`` flag the RTO loss keys on.
    # CRITICAL FIX: Cast to integer (1 or 0) rather than boolean (True/False).
    # Standard HF data collators often crash when trying to batch lists of booleans.
    rto_rows = ("mle_prefix", "rto") if legacy else ("rto",)
    for row in rows:
        if "safe" not in row:
            row["safe"] = int(row.get("augmentation") in rto_rows)

    return rows


class DeRTaHFTrainer(SFTTrainer if _TRL_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """:class:`trl.SFTTrainer` implementing the full DeRTa objective.

    Component (1), MLE with Harmful Response Prefix, is the data-level
    augmentation applied by :func:`prepare_derta_dataset` (run automatically
    when ``raw_examples`` is supplied).

    Component (2), Reinforced Transition Optimization, is implemented here in
    :meth:`compute_loss`: for every batch element flagged ``safe`` a second
    cross-entropy loss is taken against a label stream in which every
    response token has been replaced by the transition-to-refusal token id.
    This per-token transition loss is added to the standard MLE loss.

    The RTO term fires whenever the batch carries a ``safe`` mask (added by
    :func:`prepare_derta_dataset`).  Batches without it degrade gracefully to
    plain SFT, so the trainer is safe to use with arbitrary datasets.
    """

    def __init__(
        self,
        *args: Any,
        raw_examples: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> None:
        if _TRL_IMPORT_ERROR is not None:
            raise ImportError(
                "trl is required for DeRTaHFTrainer"
            ) from _TRL_IMPORT_ERROR
        if _DERTA_IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.data_compiler.derta is unavailable"
            ) from _DERTA_IMPORT_ERROR

        if raw_examples is not None and "train_dataset" not in kwargs:
            kwargs["train_dataset"] = prepare_derta_dataset(raw_examples)

        super().__init__(*args, **kwargs)
        
        # CRITICAL FIX: Prevent Hugging Face Trainer from stripping the custom RTO flags.
        # Trainer._remove_unused_columns defaults to True and will silently drop "safe"
        # because it is not in the model's forward() signature.
        if getattr(self, "_signature_columns", None) is not None:
            for key in ["safe", "safe_labels", "binary_safe"]:
                if key not in self._signature_columns:
                    self._signature_columns.append(key)

        self._derta_enable_rto = bool(getattr(self.args, "enable_rto", True))
        self._derta_rto_weight = float(getattr(self.args, "rto_weight", 1.0))
        self._derta_refusal_token_id = getattr(self.args, "rto_refusal_token_id", None)
        self._derta_refusal_text = str(getattr(self.args, "rto_refusal_text", RTO_REFUSAL_TEXT))
        # Resolved lazily on first use (needs the tokenizer / model).
        self._derta_resolved_token_id: Optional[int] = None
        self._derta_legacy = getattr(self.args, "legacy_derta", None)
        if self._derta_legacy is None:
            from safetune.config import get_config
            self._derta_legacy = get_config().legacy_derta
        if not self._derta_legacy and self._derta_rto_weight != 1.0:
            import warnings
            warnings.warn("DeRTa: rto_weight only applies to the legacy two-term loss "
                          "(legacy_derta=True); the authors' objective is one "
                          "cross-entropy over both row types", stacklevel=2)

    # ------------------------------------------------------------------
    # RTO helpers
    # ------------------------------------------------------------------
    def _resolve_refusal_token(self, model: Any) -> int:
        """Return the transition-to-refusal token id.

        Uses the explicit ``rto_refusal_token_id`` config field when set.
        Otherwise takes the first token of ``rto_refusal_text`` ("Sorry") under
        the trainer's tokenizer, which is the authors' 19701 on Llama-3.
        Raises when there is no tokenizer to derive it from, or when the id is
        outside the model's vocabulary.
        """
        if self._derta_resolved_token_id is not None:
            return self._derta_resolved_token_id

        token_id = self._derta_refusal_token_id
        tokenizer = getattr(self, "processing_class", None) or getattr(
            self, "tokenizer", None
        )

        if token_id is None:
            if tokenizer is None:
                raise ValueError("DeRTa RTO needs a tokenizer (processing_class) "
                                 "or an explicit rto_refusal_token_id")
            token_id = refusal_token_id(tokenizer, self._derta_refusal_text)

        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if vocab_size is not None and not (0 <= token_id < int(vocab_size)):
            raise ValueError(f"DeRTa refusal token id {token_id} is outside the "
                             f"model vocabulary ({vocab_size})")

        self._derta_resolved_token_id = int(token_id)
        return self._derta_resolved_token_id

    @staticmethod
    def _get_safe_mask(inputs: Any) -> Optional[Any]:
        """Extract the per-example binary ``safe`` flag from the batch.

        Mirrors ``binary_safe`` in the authors' ``MyLlamaForCausalLM.forward``.
        Returns ``None`` when the batch carries no safe flag (-> plain SFT).
        """
        if not isinstance(inputs, dict):
            return None
        for key in ("safe", "safe_labels", "binary_safe"):
            if key in inputs:
                return inputs[key]
        return None

    def _rto_transition_loss(self, model: Any, inputs: dict, logits: Any) -> Any:
        """Per-token Reinforced Transition Optimization loss.

        For every batch element flagged ``safe`` the label stream is rebuilt
        so that **every** non-ignored response token becomes the
        transition-to-refusal token id, and a next-token cross-entropy is
        taken against it (faithful to ``MyLlamaForCausalLM.forward``).
        Elements not flagged ``safe`` contribute no RTO loss.
        """
        import torch
        import torch.nn as nn

        labels = inputs.get("labels")
        if labels is None or logits is None:
            return None

        safe_mask = self._get_safe_mask(inputs)
        if safe_mask is None:
            return None

        # Normalise the safe mask to a 1-D boolean tensor, one value per row.
        if not torch.is_tensor(safe_mask):
            safe_mask = torch.as_tensor(safe_mask, device=labels.device)
        safe_mask = safe_mask.reshape(safe_mask.shape[0]).to(torch.bool)
        if not bool(safe_mask.any()):
            return None

        refusal_id = self._resolve_refusal_token(model)

        # Build safe_labels: copy of labels with every response token (where
        # label != -100) replaced by the refusal-transition token, but only
        # for the rows flagged ``safe``.  Non-safe rows stay -100 everywhere
        # so they contribute nothing to the RTO term.
        safe_labels = labels.clone()
        response_mask = labels != -100                       # (B, T)
        row_safe = safe_mask.view(-1, 1).to(response_mask.device)
        transition_targets = response_mask & row_safe        # (B, T)
        safe_labels[transition_targets] = refusal_id
        safe_labels[~transition_targets] = -100

        if not bool((safe_labels != -100).any()):
            return None

        # Standard next-token shift + cross-entropy against safe_labels.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_safe_labels = safe_labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        rto_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_safe_labels.view(-1).to(shift_logits.device),
        )
        return rto_loss

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _relabel_rto_rows(self, model: Any, inputs: dict) -> dict:
        """The authors' ``safe_labels``: on rows flagged ``safe`` every response
        token's label becomes the refusal token; other rows keep their labels."""
        import torch

        safe = self._get_safe_mask(inputs)
        labels = inputs.get("labels")
        if safe is None or labels is None:
            return inputs
        rows = torch.as_tensor(safe, device=labels.device).reshape(-1).bool().view(-1, 1)
        labels = labels.clone()
        labels[rows & (labels != -100)] = self._resolve_refusal_token(model)
        return {**inputs, "labels": labels}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        """DeRTa loss.

        Default: the authors' single cross-entropy (``MyLlamaForCausalLM``):
        MLE rows train on the safe response only (their harmful prefix is
        masked by the data), RTO rows on the refusal token at every position.
        ``legacy_derta=True``: the base MLE loss over every labelled token plus
        ``rto_weight`` x a separate RTO loss.
        """
        if self._derta_enable_rto and not self._derta_legacy:
            inputs = self._relabel_rto_rows(model, inputs)
            try:
                return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                            num_items_in_batch=num_items_in_batch)
            except TypeError:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)
        try:
            result = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        except TypeError:
            result = super().compute_loss(model, inputs, return_outputs=True)

        loss, outputs = result

        if self._derta_enable_rto:
            logits = None
            if isinstance(outputs, dict):
                logits = outputs.get("logits")
            else:
                logits = getattr(outputs, "logits", None)

            rto_loss = self._rto_transition_loss(model, inputs, logits)
            if rto_loss is not None:
                # Cast added to ensure mixed precision (bf16/fp16) safety when adding
                loss = loss + (self._derta_rto_weight * rto_loss).to(loss.dtype)

        return (loss, outputs) if return_outputs else loss

from ._deprecated import renamed as _renamed

__getattr__ = _renamed(__name__, DeRTaTrainer="DeRTaHFTrainer")  # old name, remove in 0.3
