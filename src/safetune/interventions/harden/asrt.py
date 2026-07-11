"""
ASRT: Autonomous Self-Red-Teaming training-time callback.

ASRT wraps any HuggingFace ``Trainer`` (or any of SafeTune's HARDEN
trainers) with a periodic red-teaming loop:

  1. At the end of every ``eval_steps`` interval, the callback picks the
     next ``probes_per_eval`` adversarial prompts from a rotating queue.
  2. It generates a completion with the current model (via ``attacker``).
  3. A judge (default: :class:`StringMatchJudge`) scores each completion.
  4. Compliant responses (i.e. the model failed to refuse) are converted
     to *contrastive training examples* of the form ``(prompt, refusal)``
     and buffered.
  5. On :meth:`on_epoch_end` the buffered rows are appended to the host
     trainer's ``train_dataset`` so the *next* epoch trains on the holes
     the red-team found.

In plain terms: the trainer keeps a small attacker on standby. Whenever the
attacker finds a hole, the trainer remembers it and trains the model to
patch it in the next pass.

Provenance / honesty note
-------------------------
``ASRT`` is **not** a published method and there is no "canonical paper" for
it -- the name is SafeTune-internal. It is a SafeTune-original heuristic in
the *multi-round automatic red-teaming* lineage. The closest published work
is **MART** ("Improving LLM Safety with Multi-round Automatic Red-Teaming",
Ge et al., arXiv:2311.07689).

**Why ASRT is not MART and cannot be made into MART without a full rewrite:**
MART requires (a) *two separately trained LLMs* — an adversarial LLM M_adv
that is itself SFT-trained each round on successful attacks, and a target LLM
M_tgt; (b) *two pre-trained reward models* (one for safety, one for
helpfulness) that filter M_tgt's sampled responses; (c) a round-wise outer
loop in which both M_adv and M_tgt are retrained.  None of this fits the
``TrainerCallback`` model: a single callback cannot host two trainable models,
run SFT on M_adv each round, or replace the prompt pool with RM-filtered
generations.  ASRT implements the same *discover-judge-retrain* goal with a
single model, a static prompt pool, and a string-match judge — which is
simpler and deployable but structurally different from MART.

ASRT is *inspired by* but does **not implement** Latent Adversarial Training
(Casper et al., arXiv:2403.05030; Sheshadri et al., arXiv:2407.15549). LAT
perturbs latent activations in the residual stream; ASRT performs no latent
perturbation -- it augments the *training data* with discovered
prompt/refusal pairs. The earlier reference to LAT as a "canonical paper"
was an over-claim and has been removed.

Implementation notes:

* The attacker is *any* object exposing ``.generate(prompts) -> List[str]``
  (or a SafeTune ``InferenceBackend`` reachable via ``.backend.generate``).
  For a stronger attack, pass a :class:`safetune.evaluate.redteam.BoNAttack`
  wrapper or an external red-team harness.
* The "refusal target" is a single configurable canned string
  (``ASRTConfig.refusal_target``). Every discovered failure maps to that
  same label. For per-prompt targets (e.g. a context-aware refusal),
  inject a custom ``contrastive_builder``.
* Dataset append needs a reference to the host trainer. HuggingFace does
  *not* pass the trainer into callback hooks, so either pass ``trainer=``
  at construction or call :meth:`bind_trainer` before ``trainer.train()``.
  If neither is done the callback still red-teams and buffers, but logs a
  warning instead of silently losing the discovered examples.
* If ``train_dataset`` is a ``datasets.Dataset`` the rows are appended via
  ``concatenate_datasets`` (single rebuild); if it is a list of dicts they
  are extended in place. Other types are logged and the buffer is kept
  intact rather than dropped.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


try:
    from transformers import TrainerCallback
    _TC_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    TrainerCallback = object  # type: ignore[assignment,misc]
    _TC_IMPORT_ERROR = _e


@dataclass
class ASRTConfig:
    """Configuration for the ASRT training-time callback.

    Attributes:
        eval_steps: run a red-team pass every N global training steps.
            ``200`` matches the HF default ``logging_steps``.
        probes_per_eval: how many adversarial prompts to issue per eval round.
        max_new_tokens: cap on attacker / model generations.
        refusal_target: the canned response inserted as the contrastive
            target for every successful attack.
        max_buffer: keep at most this many successful attacks in the
            contrastive buffer (rolling FIFO).
        early_stop_at_zero_asr: stop appending new examples once an entire
            eval round produces zero successful attacks. ``False`` by
            default so the callback keeps probing for regressions.
    """

    eval_steps: int = 200
    probes_per_eval: int = 8
    max_new_tokens: int = 128
    refusal_target: str = "I cannot help with that request."
    max_buffer: int = 256
    early_stop_at_zero_asr: bool = False


class ASRTCallback(TrainerCallback if _TC_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """Trainer callback that red-teams the model during training.

    Hooks:
      * :meth:`on_step_end`: increments a step counter; every
        ``eval_steps`` runs a red-team pass on ``probes_per_eval`` prompts.
      * :meth:`on_epoch_end`: appends contrastive (prompt, refusal) pairs
        accumulated since the last epoch into ``trainer.train_dataset``.

    Example::

        from safetune.core.eval.pipeline.backends import TransformersBackend
        from safetune.harden.asrt import ASRTCallback, ASRTConfig

        attacker = TransformersBackend(model="meta-llama/Llama-3.2-1B-Instruct")
        callback = ASRTCallback(
            attacker=attacker,
            adversarial_prompts=load_prompts("harmbench")[:64],
            config=ASRTConfig(eval_steps=100, probes_per_eval=4),
        )
        trainer = SFTTrainer(..., callbacks=[callback])
        callback.bind_trainer(trainer)   # required for the epoch-end append
        trainer.train()

    Note that the buffered contrastive rows are appended on ``on_epoch_end``
    and are therefore only trained on by *subsequent* epochs. With a
    single-epoch fine-tuning recipe the discovered examples are appended but
    never trained on -- use ``num_train_epochs >= 2`` for ASRT to have an
    effect on the model.
    """

    def __init__(
        self,
        attacker: Any,
        adversarial_prompts: Sequence[str],
        judge: Optional[Any] = None,
        config: Optional[ASRTConfig] = None,
        contrastive_builder: Optional[Callable[[str], Dict[str, Any]]] = None,
        trainer: Optional[Any] = None,
    ) -> None:
        """Create the callback.

        Args:
            attacker: object exposing ``.generate(prompts) -> List[str]``
                (or ``.backend.generate``); used to produce model
                completions of the adversarial prompts.
            adversarial_prompts: non-empty pool of attack prompts; cycled
                round-robin across red-team rounds.
            judge: scorer with ``.score(rows) -> rows`` adding a
                ``judgement.asr`` field. Defaults to ``StringMatchJudge``.
            config: :class:`ASRTConfig`; defaults are used if omitted.
            contrastive_builder: optional ``prompt -> row`` factory for
                custom (e.g. per-prompt) refusal targets. Defaults to a
                single canned-refusal builder.
            trainer: optional host ``Trainer``. Required for the
                ``on_epoch_end`` dataset append to take effect; if not
                given here it can be supplied later via
                :meth:`bind_trainer`. HuggingFace does not pass the
                trainer into callback hooks, so one of the two is needed.
        """
        if _TC_IMPORT_ERROR is not None:
            raise ImportError("transformers is required for ASRTCallback") from _TC_IMPORT_ERROR
        if not adversarial_prompts:
            raise ValueError("ASRTCallback: ``adversarial_prompts`` cannot be empty.")

        from safetune.core.eval.pipeline.scorer import StringMatchJudge  # lazy import

        self.attacker = attacker
        self.adversarial_prompts: Deque[str] = deque(adversarial_prompts)
        self.judge = judge if judge is not None else StringMatchJudge()
        self.config = config or ASRTConfig()
        self._step_counter = 0
        self._buffer: List[Dict[str, Any]] = []
        self._contrastive_builder = contrastive_builder or self._default_contrastive_builder
        self._zero_asr_seen = False
        self._trainer = trainer
        self._warned_no_trainer = False

    def bind_trainer(self, trainer: Any) -> "ASRTCallback":
        """Attach the host trainer so :meth:`on_epoch_end` can append rows.

        Call this before ``trainer.train()`` if the trainer was not passed
        to ``__init__``. Returns ``self`` for chaining.
        """
        self._trainer = trainer
        return self

    def _default_contrastive_builder(self, prompt: str) -> Dict[str, Any]:
        """Build one (prompt, refusal) row in HF-Trainer-compatible shape."""
        return {
            "prompt": prompt,
            "completion": self.config.refusal_target,
            "asrt_origin": "successful_attack",
        }

    def _generate(self, prompts: List[str]) -> List[str]:
        a = self.attacker
        if hasattr(a, "backend") and hasattr(a.backend, "generate"):
            return a.backend.generate(prompts)
        if hasattr(a, "generate"):
            return a.generate(prompts)
        raise TypeError("ASRTCallback: attacker must have a .generate(prompts) method.")

    def _redteam_round(self) -> int:
        """Run one red-team round; return number of successful attacks found."""
        if not self.adversarial_prompts:
            return 0
        n = min(self.config.probes_per_eval, len(self.adversarial_prompts))
        batch_prompts = [self.adversarial_prompts.popleft() for _ in range(n)]
        # Rotate them back so the queue cycles indefinitely.
        for p in batch_prompts:
            self.adversarial_prompts.append(p)

        responses = self._generate(batch_prompts)
        rows = [{"prompt": p, "response": r} for p, r in zip(batch_prompts, responses)]
        judged = self.judge.score(rows)
        successes = 0
        for row in judged:
            if row.get("judgement", {}).get("asr", 0.0) >= 1.0:
                self._buffer.append(self._contrastive_builder(row["prompt"]))
                successes += 1
        # FIFO trim.
        if len(self._buffer) > self.config.max_buffer:
            self._buffer = self._buffer[-self.config.max_buffer :]
        logger.info("ASRT: red-team round found %d/%d successful attacks.", successes, len(rows))
        return successes

    # ----------------------------------------------------------------- hooks

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self._step_counter += 1
        if self._step_counter % max(1, self.config.eval_steps) != 0:
            return control
        if self.config.early_stop_at_zero_asr and self._zero_asr_seen:
            return control
        successes = self._redteam_round()
        self._zero_asr_seen = successes == 0
        return control

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if not self._buffer:
            return control
        # HuggingFace does not pass the trainer into callback kwargs; we use
        # the explicitly bound reference, with a best-effort fallback to a
        # ``model.trainer`` back-pointer some setups attach. An explicit
        # ``trainer=`` kwarg (the pre-fix call convention) is also honoured for
        # backward compatibility.
        trainer = (
            self._trainer
            or kwargs.get("trainer")
            or getattr(kwargs.get("model"), "trainer", None)
        )
        if trainer is None:
            if not self._warned_no_trainer:
                logger.warning(
                    "ASRT: no trainer bound; %d discovered contrastive rows "
                    "cannot be appended to train_dataset. Pass trainer= to "
                    "ASRTCallback(...) or call .bind_trainer(trainer) before "
                    "trainer.train(). Buffer kept intact.",
                    len(self._buffer),
                )
                self._warned_no_trainer = True
            return control
        # Best-effort dataset append. Trainer subclasses sometimes hide the
        # dataset behind a property; we degrade to logging rather than
        # raising or corrupting state.
        appended = self._append_to_train_dataset(trainer)
        logger.info(
            "ASRT: epoch end, appended %d contrastive rows (buffer size %d).",
            appended, len(self._buffer),
        )
        return control

    def _append_to_train_dataset(self, trainer: Any) -> int:
        if trainer is None or not hasattr(trainer, "train_dataset"):
            return 0
        ds = trainer.train_dataset
        if ds is None:
            return 0
        if isinstance(ds, list):
            # Plain list of dict rows: extend in place.
            count = len(self._buffer)
            ds.extend(self._buffer)
            self._buffer.clear()
            return count
        if hasattr(ds, "add_item"):
            # HuggingFace ``datasets.Dataset``: build the new rows into a
            # single Dataset and concatenate once (O(n), not O(n^2)), and
            # only clear the buffer if the rebuild fully succeeds so a
            # failure does not silently drop discovered examples.
            try:
                from datasets import Dataset, concatenate_datasets

                addition = Dataset.from_list(list(self._buffer))
                trainer.train_dataset = concatenate_datasets([ds, addition])
            except Exception as e:  # pragma: no cover - depends on schema
                logger.warning(
                    "ASRT: failed to append %d rows to datasets.Dataset "
                    "(%s); buffer kept intact.",
                    len(self._buffer), e,
                )
                return 0
            count = len(self._buffer)
            self._buffer.clear()
            return count
        logger.warning(
            "ASRT: train_dataset type %s not supported for append; "
            "leaving contrastive buffer intact.",
            type(ds).__name__,
        )
        return 0

    # --------------------------------------------------------------- helpers

    @property
    def buffer(self) -> List[Dict[str, Any]]:
        """Read-only view of the contrastive buffer (test introspection)."""
        return list(self._buffer)


__all__ = ["ASRTCallback", "ASRTConfig"]
