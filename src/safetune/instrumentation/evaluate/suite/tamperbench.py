"""TamperBench evaluator — thin wrapper over the official TamperBench harness.

Reference
---------
TamperBench: "TamperBench: Systematically Stress-Testing LLM Safety Under
Fine-Tuning and Tampering", Hossain, Tseng, Pandey, Vajpayee, Kowal, Nonta,
Simko, Casper et al., arXiv:2602.06911 (2026).
Official code: https://github.com/criticalml-uw/TamperBench  (package ``tamperbench``).

What TamperBench actually is
----------------------------
TamperBench is the unified *tamper-resistance* benchmark for open-weight LLMs.
It is **not** a GPT-4 1-5 harmfulness judge — that is the separate Qi et al.
2023 ("Fine-tuning Aligned Language Models Compromises Safety", arXiv:2310.03693)
judge. The previous SafeTune implementation in ``safetune.core.eval.tamperbench``
mis-cited that GPT-4 judge as "TamperBench"; this module replaces that
re-export with a faithful thin wrapper over the real harness.

The real TamperBench protocol (from the paper + repo) is:

* **Tampering attacks** (``tamperbench.whitebox.utils.names.AttackName``):
  weight-space fine-tuning attacks — ``full_parameter_finetune``,
  ``lora_finetune``, ``benign_full_parameter_finetune``, ``benign_lora_finetune``,
  ``multilingual_finetune``, ``competing_objectives_finetune``,
  ``style_modulation_finetune``, ``backdoor_finetune`` (the jailbreak-tuning
  family), ``refusal_ablation`` — and the latent/representation-space
  ``embedding_attack``, plus ``prompt_attack`` / ``gcg_attack`` and the
  ``no_weight_modification`` baseline.
* **Safety metric**: each candidate (post-attack) model is run on the
  StrongREJECT forbidden-prompt set; responses are scored by the StrongREJECT
  *fine-tuned* evaluator (``strong_reject.evaluate.strongreject_finetuned``,
  a fine-tuned Gemma-2B), which returns a harmfulness score in ``[0, 1]``
  (higher = more harmful, accounting for refusal / specificity / convincingness).
  The reported metric is the **mean** StrongREJECT score
  (``MetricName.STRONG_REJECT_SCORE``). Attackers maximize it; defenders
  minimize it. Across an attack grid TamperBench aggregates with
  ``worst_case`` (``SRmax``) or top-N-average (``SRmal-avg``).
* **Utility metric**: MMLU-Pro accuracy (``mmlu_pro_val`` — a 140-sample
  5-shot CoT subset); attacks are constrained to <=10% MMLU-Pro drop.
* **Harness**: decorator-based attack/eval registries
  (``ATTACKS_REGISTRY``, ``EVALS_REGISTRY``) plus Optuna hyper-parameter
  sweeps; the ``attack -> train -> evaluate`` pipeline runs on GPU via vLLM.

Faithfulness notes
------------------
* Running the full TamperBench pipeline requires the ``tamperbench`` package
  (vLLM, GPU). This wrapper does **not** re-implement any attack or metric; it
  delegates entirely to the real harness. When ``tamperbench`` is not
  installed, the methods raise a clear ``ImportError`` instead of silently
  computing a non-TamperBench number.
* :meth:`score_responses` is a convenience entry point that scores
  externally-supplied ``(prompt, response)`` pairs with the *real*
  StrongREJECT fine-tuned evaluator (the exact safety metric TamperBench
  uses). It needs only the ``strong_reject`` package, not GPU training.
* The legacy ``judge_responses`` name and the ``openai_api_key`` constructor
  argument are kept as deprecated aliases so existing callers do not break;
  ``openai_api_key`` is now unused (TamperBench's metric is the StrongREJECT
  classifier, not an OpenAI GPT-4 judge).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

__all__ = ["TamperBenchEvaluator"]

# Default StrongREJECT-classifier truncation length used by TamperBench's
# StrongReject eval (the strong_reject fine-tuned Gemma-2B context window).
_DEFAULT_MAX_RESPONSE_LENGTH = 512

# The tampering attacks shipped by the official TamperBench harness
# (tamperbench.whitebox.utils.names.AttackName). Kept here only so callers can
# discover valid attack names without importing the GPU package; the actual
# attack implementations live entirely in `tamperbench`.
_TAMPERBENCH_ATTACKS = (
    "full_parameter_finetune",
    "lora_finetune",
    "benign_full_parameter_finetune",
    "benign_lora_finetune",
    "multilingual_finetune",
    "competing_objectives_finetune",
    "style_modulation_finetune",
    "backdoor_finetune",
    "embedding_attack",
    "refusal_ablation",
    "prompt_attack",
    "gcg_attack",
    "no_weight_modification",
)

# TamperBench's standardized safety + utility evaluations
# (tamperbench.whitebox.utils.names.EvalName).
_TAMPERBENCH_EVALS = (
    "strong_reject",
    "jailbreak_bench",
    "mmlu_pro_val",
    "mmlu_pro",
    "mt_bench",
    "mbpp",
    "minerva_math",
    "ifeval",
    "embedding_attack_eval",
    "policy_eval",
    "xstest",
    "wmdp",
)


def _harness_unavailable_error(exc: Optional[BaseException]) -> ImportError:
    """Build a clear ImportError explaining the TamperBench harness is missing."""
    msg = (
        "The official TamperBench harness (`tamperbench`, arXiv:2602.06911, "
        "https://github.com/criticalml-uw/TamperBench) is not installed. "
        "TamperBenchEvaluator is a thin wrapper and does not re-implement the "
        "benchmark. Install it (`pip install` from the repo / `uv sync`) to run "
        "tampering attacks and TamperBench evaluations."
    )
    return ImportError(msg) if exc is None else ImportError(msg)


class TamperBenchEvaluator:
    """Thin wrapper over the official TamperBench tamper-resistance harness.

    This class does not implement any tampering attack or metric itself — it
    delegates to the ``tamperbench`` package (arXiv:2602.06911). It can be
    constructed on CPU with no dependencies; the heavy ``tamperbench`` /
    ``strong_reject`` imports happen lazily inside the methods so that simply
    importing :class:`TamperBenchEvaluator` never requires a GPU.

    Two entry points are provided:

    * :meth:`run_attack` — drive the full TamperBench ``attack -> train ->
      evaluate`` pipeline for one tampering attack via the real harness.
    * :meth:`score_responses` (alias :meth:`judge_responses`) — score
      already-generated ``(prompt, response)`` pairs with TamperBench's actual
      safety metric: the StrongREJECT fine-tuned evaluator.
    """

    #: Tampering attacks supported by the official TamperBench harness.
    SUPPORTED_ATTACKS: Tuple[str, ...] = _TAMPERBENCH_ATTACKS
    #: Evaluations supported by the official TamperBench harness.
    SUPPORTED_EVALS: Tuple[str, ...] = _TAMPERBENCH_EVALS

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        *,
        model_checkpoint: Optional[str] = None,
        out_dir: str = "tamperbench_results",
        evals: Optional[Sequence[str]] = None,
        random_seed: int = 0,
        max_response_length: int = _DEFAULT_MAX_RESPONSE_LENGTH,
    ) -> None:
        """Initialize the TamperBench wrapper.

        Args:
            openai_api_key: **Deprecated and unused.** Kept only so existing
                callers do not break. TamperBench's safety metric is the
                StrongREJECT fine-tuned classifier, not an OpenAI GPT-4 judge,
                so no API key is needed. Passing a non-None value emits a
                ``DeprecationWarning``.
            model_checkpoint: HuggingFace path / local path of the (aligned)
                model to tamper-test. Required only for :meth:`run_attack`.
            out_dir: Directory where TamperBench writes checkpoints, inferences
                and evaluation artifacts.
            evals: TamperBench evaluation names to run after an attack
                (default: ``("strong_reject", "mmlu_pro_val")`` — the paper's
                safety + utility pair). Must be a subset of
                :attr:`SUPPORTED_EVALS`.
            random_seed: Random seed forwarded to the harness.
            max_response_length: Response-truncation length for the
                StrongREJECT fine-tuned classifier (its context window;
                TamperBench / ``strong_reject`` default is 512).
        """
        if openai_api_key is not None:
            warnings.warn(
                "`openai_api_key` is deprecated and ignored: TamperBench scores "
                "safety with the StrongREJECT fine-tuned classifier, not an "
                "OpenAI GPT-4 judge.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.model_checkpoint = model_checkpoint
        self.out_dir = out_dir
        self.random_seed = int(random_seed)
        self.max_response_length = int(max_response_length)

        if evals is None:
            evals = ("strong_reject", "mmlu_pro_val")
        unknown = [e for e in evals if e not in self.SUPPORTED_EVALS]
        if unknown:
            raise ValueError(
                f"Unknown TamperBench eval(s) {unknown!r}; "
                f"valid names: {self.SUPPORTED_EVALS}"
            )
        self.evals: Tuple[str, ...] = tuple(evals)

        # Resolved lazily; None until first use.
        self._tamperbench: Any = None
        self._import_error: Optional[BaseException] = None

    # ------------------------------------------------------------------ #
    # Lazy harness resolution
    # ------------------------------------------------------------------ #
    def _load_harness(self) -> Any:
        """Import and cache the real ``tamperbench`` package, or raise clearly."""
        if self._tamperbench is not None:
            return self._tamperbench
        try:
            import tamperbench  # type: ignore

            self._tamperbench = tamperbench
            return tamperbench
        except Exception as exc:  # pragma: no cover - depends on env
            self._import_error = exc
            raise _harness_unavailable_error(exc) from exc

    @property
    def harness_available(self) -> bool:
        """Whether the official ``tamperbench`` package is importable."""
        try:
            self._load_harness()
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------ #
    # Full pipeline: tamper attack -> train -> evaluate
    # ------------------------------------------------------------------ #
    def run_attack(
        self,
        attack: str,
        *,
        model_checkpoint: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run one TamperBench tampering attack and its evaluations.

        This delegates entirely to the official harness: it looks the attack up
        in ``tamperbench``'s ``ATTACKS_REGISTRY``, builds the attack config,
        and calls the attack's ``benchmark()`` (``attack -> train -> evaluate``).
        Requires the ``tamperbench`` package and a GPU.

        Args:
            attack: A TamperBench attack name (see :attr:`SUPPORTED_ATTACKS`),
                e.g. ``"lora_finetune"`` or ``"full_parameter_finetune"``.
            model_checkpoint: Model to tamper-test; overrides the constructor
                value for this call.
            config_overrides: Extra fields forwarded into the attack's config
                dataclass (learning rate, epochs, lora_rank, ...).

        Returns:
            The harness's structured benchmark result (per-eval TamperBench
            metrics such as ``strong_reject_score`` and ``mmlu_pro_accuracy``).

        Raises:
            ImportError: If the ``tamperbench`` package is not installed.
            ValueError: If ``attack`` is not a known TamperBench attack or no
                model checkpoint was provided.
        """
        if attack not in self.SUPPORTED_ATTACKS:
            raise ValueError(
                f"Unknown TamperBench attack {attack!r}; "
                f"valid names: {self.SUPPORTED_ATTACKS}"
            )
        checkpoint = model_checkpoint or self.model_checkpoint
        if not checkpoint:
            raise ValueError(
                "run_attack requires `model_checkpoint` (pass it to the "
                "constructor or to run_attack)."
            )

        self._load_harness()  # raises a clear ImportError if missing

        from tamperbench.whitebox.attacks.registry import ATTACKS_REGISTRY  # type: ignore
        from tamperbench.whitebox.utils.names import AttackName, EvalName  # type: ignore

        attack_name = AttackName(attack)
        if attack_name not in ATTACKS_REGISTRY:
            raise ValueError(
                f"Attack {attack!r} is not registered in this build of "
                f"tamperbench (registered: {sorted(map(str, ATTACKS_REGISTRY))})."
            )
        config_cls, attack_cls = ATTACKS_REGISTRY[attack_name]

        config_dict: Dict[str, Any] = {
            "input_checkpoint_path": checkpoint,
            "out_dir": self.out_dir,
            "evals": [EvalName(e) for e in self.evals],
            "random_seed": self.random_seed,
        }
        if config_overrides:
            config_dict.update(config_overrides)

        attack_config = config_cls.from_dict(config_dict)
        tamper_attack = attack_cls(attack_config=attack_config)
        return tamper_attack.benchmark()

    # ------------------------------------------------------------------ #
    # Safety metric: StrongREJECT fine-tuned classifier
    # ------------------------------------------------------------------ #
    def score_responses(
        self,
        qa_pairs: Sequence[Tuple[str, str]],
        *,
        batch_size: int = 32,
    ) -> Dict[str, Union[float, List[Dict[str, Any]]]]:
        """Score ``(prompt, response)`` pairs with TamperBench's safety metric.

        Uses the StrongREJECT *fine-tuned* evaluator
        (``strong_reject.evaluate.strongreject_finetuned``) — the exact safety
        scorer TamperBench's ``StrongRejectEvaluation`` uses. Each pair gets a
        harmfulness score in ``[0, 1]`` (higher = more harmful, accounting for
        refusal / specificity / convincingness). The reported aggregate is the
        mean score, matching ``MetricName.STRONG_REJECT_SCORE``.

        This needs only the ``strong_reject`` package (which loads a fine-tuned
        Gemma-2B classifier); it does not run any tampering attack.

        Args:
            qa_pairs: Sequence of ``(forbidden_prompt, model_response)`` tuples.
            batch_size: Classifier batch size.

        Returns:
            Dict with ``strong_reject_score`` (mean StrongREJECT harmfulness in
            ``[0, 1]`` — the TamperBench safety metric), ``num_evaluated`` (number
            of pairs scored), and ``results`` (per-pair
            ``{prompt, response, score}`` records).

        Raises:
            ImportError: If the ``strong_reject`` package is not installed.
        """
        pairs = list(qa_pairs)
        if not pairs:
            return {"strong_reject_score": 0.0, "num_evaluated": 0, "results": []}

        try:
            from strong_reject.evaluate import (  # type: ignore
                strongreject_finetuned,
            )
        except Exception as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "TamperBench scores safety with the StrongREJECT fine-tuned "
                "evaluator, which requires the `strong_reject` package "
                "(https://github.com/dsbowen/strong_reject). Install it to use "
                "score_responses()."
            ) from exc

        scores: List[float] = []
        for start in range(0, len(pairs), max(1, batch_size)):
            chunk = pairs[start : start + max(1, batch_size)]
            batch = {
                "forbidden_prompt": [p for p, _ in chunk],
                "response": [r for _, r in chunk],
            }
            out = strongreject_finetuned(
                batch=batch,
                max_response_length=self.max_response_length,
            )
            batch_scores = out.get("score")
            if batch_scores is None:  # pragma: no cover - defensive
                raise RuntimeError(
                    "strongreject_finetuned did not return a 'score' field."
                )
            scores.extend(float(s) for s in batch_scores)

        results = [
            {"prompt": p, "response": r, "score": s}
            for (p, r), s in zip(pairs, scores)
        ]
        mean_score = sum(scores) / len(scores)
        return {
            "strong_reject_score": mean_score,
            "num_evaluated": len(scores),
            "results": results,
        }

    # ------------------------------------------------------------------ #
    # Backward-compatible alias
    # ------------------------------------------------------------------ #
    def judge_responses(
        self,
        qa_pairs: Sequence[Tuple[str, str]],
        *,
        batch_size: int = 32,
    ) -> Dict[str, Union[float, List[Dict[str, Any]]]]:
        """Deprecated alias for :meth:`score_responses`.

        The previous implementation called an OpenAI GPT-4 1-5 judge; that is
        the Qi et al. 2023 judge, not the TamperBench metric. This now routes
        to the real TamperBench safety scorer (StrongREJECT fine-tuned).
        """
        warnings.warn(
            "judge_responses is deprecated; use score_responses. It now scores "
            "with TamperBench's actual safety metric (StrongREJECT fine-tuned), "
            "not the Qi et al. 2023 GPT-4 1-5 judge.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.score_responses(qa_pairs, batch_size=batch_size)
