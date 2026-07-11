"""
Judge backend adapters for safety scoring.

This module is the **single source of truth** for the judge-adapter
abstraction. Adapter classes encapsulate how different judge backends turn a
list of texts into scalar safety scores, exposing a uniform contract::

    score(texts: List[str]) -> List[float]

so that callers such as ``score_with_backend_checked`` (``metrics.safety``) and
the Llama-Guard scoring path (``verify.eval.evaluate``) stay small and
backend-agnostic.

Backends
--------
* ``local``               -- a user-supplied ``judge_fn`` with a fallback.
* ``classifier``          -- a per-text classifier callable.
* ``batched_classifier``  -- a callable that consumes a whole batch at once.
* ``api``                 -- a per-text API callable with timeout logic.
* ``judge``               -- delegate to a *corrected* judge object from
  :mod:`safetune.core.eval.pipeline.scorer` (``StringMatchJudge`` / ``HFJudge`` /
  ``OpenAIJudge``). This is the bridge that lets the row-based, benchmark-grade
  judges in ``scorer.py`` be used behind the flat adapter contract.

``verify/eval/evaluate.py`` and ``verify/eval/judge.py`` re-export the names
defined here; they do **not** redefine them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol


class JudgeAdapter(Protocol):
    """Protocol for judge adapters."""

    def score(self, texts: List[str]) -> List[float]:  # pragma: no cover - structural
        ...


@dataclass
class LocalJudgeAdapter:
    """Local judge adapter.

    Uses a user-supplied ``judge_fn`` if provided; otherwise defers to a
    fallback function (typically the module-level harmfulness heuristic).
    """

    judge_fn: Optional[Callable[[str], float]]
    fallback_fn: Callable[[str], float]

    def score(self, texts: List[str]) -> List[float]:
        if self.judge_fn is not None:
            return [float(self.judge_fn(t)) for t in texts]
        return [float(self.fallback_fn(t)) for t in texts]


@dataclass
class ClassifierJudgeAdapter:
    """Classifier-based judge adapter.

    Wraps a high-throughput safety classifier ``fn(text) -> score``.
    """

    classifier_fn: Callable[[str], float]

    def score(self, texts: List[str]) -> List[float]:
        return [float(self.classifier_fn(t)) for t in texts]


@dataclass
class BatchedClassifierJudgeAdapter:
    """Optimized adapter that passes full batches directly to the scorer.

    Wraps a classifier that consumes a whole batch at once
    (``fn(texts) -> scores``) -- e.g. a batched HF-generation judge -- instead
    of being called once per text.
    """

    batched_classifier_fn: Callable[[List[str]], List[float]]

    def score(self, texts: List[str]) -> List[float]:
        return [float(s) for s in self.batched_classifier_fn(texts)]


@dataclass
class APIJudgeAdapter:
    """API-based judge adapter.

    The ``timeout_fn`` is expected to implement any network timeout logic
    and return a scalar score for each text.
    """

    timeout_fn: Callable[[str], float]

    def score(self, texts: List[str]) -> List[float]:
        return [float(self.timeout_fn(t)) for t in texts]


@dataclass
class ScorerJudgeAdapter:
    """Adapter that delegates to a corrected judge from :mod:`scorer`.

    The benchmark-grade judges in :mod:`safetune.core.eval.pipeline.scorer`
    (``StringMatchJudge``, ``HFJudge``, ``OpenAIJudge``) operate on a
    *row* contract -- ``score(rows: List[Dict]) -> List[Dict]`` where each row
    carries ``prompt`` / ``response`` and the verdict is returned under a
    ``judgement`` field. This adapter bridges that row contract to the flat
    ``score(texts) -> List[float]`` adapter surface.

    Each input text is treated as a model *response*. An optional matching
    ``behaviors`` list supplies the originating prompt/behavior for judges that
    need it (``HFJudge`` / ``OpenAIJudge``); when omitted the behavior is left
    empty. The per-row ``judgement["asr"]`` produced by the judge is returned
    as the scalar score, matching the attack-success-rate convention shared by
    every ``scorer.py`` judge.

    Parameters
    ----------
    judge:
        Any object exposing ``score(rows) -> List[dict]`` -- i.e. a
        ``StringMatchJudge`` / ``HFJudge`` / ``OpenAIJudge`` instance (or any
        future judge honouring the same contract).
    behaviors:
        Optional list of prompts/behaviors aligned with the scored ``texts``.
        Used to populate the ``prompt`` field each judge reads. When ``None``
        or shorter than ``texts`` the missing entries default to ``""``.
    score_key:
        Key read from each row's ``judgement`` dict for the scalar score.
        Defaults to ``"asr"`` (the shared convention of all ``scorer.py``
        judges).
    """

    judge: Any
    behaviors: Optional[List[str]] = None
    score_key: str = "asr"

    def score(self, texts: List[str]) -> List[float]:
        behaviors = self.behaviors or []
        rows: List[Dict[str, Any]] = []
        for i, t in enumerate(texts):
            behavior = behaviors[i] if i < len(behaviors) else ""
            rows.append({"prompt": behavior, "response": t})
        scored = self.judge.score(rows)
        return [
            float(r.get("judgement", {}).get(self.score_key, 0.0)) for r in scored
        ]


def build_judge_adapter(
    backend: str,
    *,
    judge_fn: Optional[Callable[[str], float]] = None,
    classifier_fn: Optional[Callable[[str], float]] = None,
    batched_classifier_fn: Optional[Callable[[List[str]], List[float]]] = None,
    timeout_fn: Optional[Callable[[str], float]] = None,
    judge: Optional[Any] = None,
    behaviors: Optional[List[str]] = None,
    fallback_fn: Callable[[str], float],
) -> Optional[JudgeAdapter]:
    """Factory for :class:`JudgeAdapter` instances.

    Args:
        backend: One of ``local``, ``classifier``, ``batched_classifier``,
            ``api``, ``judge`` (case-insensitive).
        judge_fn: Optional callable for the ``local`` backend.
        classifier_fn: Required for the ``classifier`` backend.
        batched_classifier_fn: Required for the ``batched_classifier`` backend.
        timeout_fn: Required for the ``api`` backend.
        judge: Required for the ``judge`` backend -- a corrected judge object
            from :mod:`safetune.core.eval.pipeline.scorer` exposing
            ``score(rows) -> List[dict]``.
        behaviors: Optional prompts aligned with the scored texts, forwarded to
            :class:`ScorerJudgeAdapter` for the ``judge`` backend.
        fallback_fn: Fallback scorer for the ``local`` backend when ``judge_fn``
            is ``None``.

    Returns:
        A judge adapter, or ``None`` when the backend's required callable /
        object is missing (so the caller can surface an ``unavailable`` state
        instead of silently mis-scoring).
    """
    key = (backend or "local").lower()

    if key == "batched_classifier":
        if batched_classifier_fn is None:
            return None
        return BatchedClassifierJudgeAdapter(batched_classifier_fn=batched_classifier_fn)

    if key == "classifier":
        if classifier_fn is None:
            return None
        return ClassifierJudgeAdapter(classifier_fn=classifier_fn)

    if key == "api":
        if timeout_fn is None:
            return None
        return APIJudgeAdapter(timeout_fn=timeout_fn)

    if key == "judge":
        if judge is None:
            return None
        return ScorerJudgeAdapter(judge=judge, behaviors=behaviors)

    # Default: local backend
    return LocalJudgeAdapter(judge_fn=judge_fn, fallback_fn=fallback_fn)


__all__ = [
    "JudgeAdapter",
    "LocalJudgeAdapter",
    "ClassifierJudgeAdapter",
    "BatchedClassifierJudgeAdapter",
    "APIJudgeAdapter",
    "ScorerJudgeAdapter",
    "build_judge_adapter",
]
