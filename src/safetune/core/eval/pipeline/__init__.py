"""
Eval pipeline: multi-backend inference + judges for OOD safety benchmarks.

Modelled on the cthetha-eval design but exposed as a Python class hierarchy
so SafeTune library users compose it programmatically. Key entry points:

* :class:`Generator`: run a model over a list of prompt dicts.
* :func:`load_prompts`: load HarmBench / AdvBench / XSTest / StrongREJECT /
  SORRY-Bench / OR-Bench / WildJailbreak by name.
* :class:`StringMatchJudge`, :class:`HFJudge`, :class:`OpenAIJudge`:
  composable judges that consume Generator outputs and emit per-row
  judgements + an aggregate ASR summary.

For capability evaluation (GSM8K, MMLU, TruthfulQA, etc.) see
:mod:`safetune.core.eval.lm_eval_runner` which wraps EleutherAI's
``lm-evaluation-harness``.
"""
from .backends import (
    GenerationConfig,
    InferenceBackend,
    DryRunBackend,
    TransformersBackend,
    make_backend,
)
from .generator import Generator
from .loaders import load_prompts, LOADERS
from .scorer import (
    StringMatchJudge,
    HFJudge,
    HFJudgeConfig,
    OpenAIJudge,
    OpenAIJudgeConfig,
    asr_summary,
)

__all__ = [
    "GenerationConfig",
    "InferenceBackend",
    "DryRunBackend",
    "TransformersBackend",
    "make_backend",
    "Generator",
    "load_prompts",
    "LOADERS",
    "StringMatchJudge",
    "HFJudge",
    "HFJudgeConfig",
    "OpenAIJudge",
    "OpenAIJudgeConfig",
    "asr_summary",
]
