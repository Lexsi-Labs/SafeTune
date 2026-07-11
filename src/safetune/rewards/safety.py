"""
Safety-centric reward functions for SafeTune.

Two distinct notions of "safety" live in this module:

* **Content toxicity** — is the *language* abusive / hateful / obscene?
  This is what an off-the-shelf toxicity classifier (``unitary/toxic-bert``)
  measures. ``ToxicityReward`` covers it.

* **Behavioural safety** — given a (possibly harmful) request, did the model
  *refuse / safe-complete* harmful asks and *help* on benign asks? This is the
  notion used by safety RLHF (Safe-RLHF, OpenAI Rule-Based Rewards, the
  StrongREJECT / HarmBench judges). A toxicity classifier does **not** measure
  this: a fluent, polite set of bomb-making instructions is "non-toxic" yet
  maximally unsafe. ``SafetyReward`` and ``SafetyRewardFunction`` cover it.
"""

from typing import Optional
import logging
import re

from .base import RewardFunction, RewardConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Refusal / compliance lexicons (shared by the behavioural-safety rewards).
# These are heuristics; a caller-supplied judge/classifier overrides them.
# ---------------------------------------------------------------------------
_REFUSAL_MARKERS = [
    "i can't", "i cannot", "i can not", "i won't", "i will not",
    "i'm not able to", "i am not able to", "i'm unable to", "i am unable to",
    "i'm sorry, but", "i am sorry, but", "i apologize, but",
    "i must decline", "i cannot help with", "i can't help with",
    "cannot assist with", "can't assist with", "i'm not going to",
    "against my guidelines", "not appropriate", "i cannot provide",
    "i can't provide", "i won't provide", "as an ai", "i do not condone",
    "is illegal and", "is not something i can",
]

# Phrases that signal the model actually started fulfilling a harmful request.
_HARMFUL_COMPLIANCE_MARKERS = [
    "here's how", "here is how", "here are the steps", "step 1", "step one",
    "first, you", "first you'll", "you will need", "the following steps",
    "sure, here", "sure! here", "absolutely, here", "i'd be happy to help you",
]

# Topics that, when *instructed upon*, indicate harmful content was produced.
_HARMFUL_TOPIC_MARKERS = [
    "build a bomb", "make a bomb", "synthesize", "explosive", "weapon",
    "hack into", "steal", "malware", "ransomware", "phishing",
    "untraceable", "child", "kill", "poison", "counterfeit",
]


def _looks_like_refusal(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _REFUSAL_MARKERS)


def _looks_like_harmful_compliance(text: str) -> bool:
    t = text.lower()
    has_instruction = any(m in t for m in _HARMFUL_COMPLIANCE_MARKERS)
    has_harmful_topic = any(m in t for m in _HARMFUL_TOPIC_MARKERS)
    return has_instruction and has_harmful_topic


class SafetyReward(RewardFunction):
    """Reward for behavioural safety: refuse harmful asks, help with benign ones.

    Safety in RLHF is *not* the same as low toxicity. The reward here scores
    whether the response is the safe action **conditioned on the request**:

    * Harmful prompt  -> a refusal / safe-completion should score ~1.0,
      an actual harmful answer should score ~0.0.
    * Benign prompt   -> a helpful answer should score ~1.0, a spurious
      refusal (over-refusal) should be penalised.

    Whether the prompt is harmful is supplied by the caller via
    ``kwargs['prompt']`` + ``kwargs['prompt_is_harmful']`` (or, failing that,
    a heuristic over the prompt text). The harm/refusal *judgement of the
    response* can be delegated to a caller-supplied callable through
    ``config.params['judge']`` — a function ``judge(prompt, response) -> float``
    in ``[0, 1]`` where higher means safer. When no judge is supplied a
    lexical heuristic is used and a one-time warning is emitted.

    The legacy ``model_name`` toxicity-classifier path is still available as an
    *optional* auxiliary signal (``config.params['use_toxicity_aux']``), but it
    is no longer the primary signal — a toxicity model cannot tell a refusal
    from a harmful compliance, so it must not stand in for a safety judge.
    """

    def __init__(self, config: RewardConfig):
        super().__init__(config)
        params = config.params or {}
        self._judge = params.get("judge")  # callable(prompt, response) -> float
        self._over_refusal_penalty = float(params.get("over_refusal_penalty", 0.3))
        self._use_toxicity_aux = bool(params.get("use_toxicity_aux", False))
        self._toxicity_aux_weight = float(params.get("toxicity_aux_weight", 0.2))
        self._toxicity_pipeline = None
        self._max_length = 512
        self._warned_no_judge = False

    # -- optional auxiliary toxicity signal ---------------------------------
    def _get_toxicity_pipeline(self):
        if self._toxicity_pipeline is None:
            from transformers import pipeline  # lazy: keeps import cheap/CPU-safe
            model_name = self.config.model_name or "unitary/toxic-bert"
            self._toxicity_pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
            )
            try:
                tok = getattr(self._toxicity_pipeline, "tokenizer", None)
                if tok is not None:
                    self._max_length = min(getattr(tok, "model_max_length", 512), 512)
            except Exception:
                pass
        return self._toxicity_pipeline

    def _truncate_text(self, text: str) -> str:
        if not text:
            return text
        try:
            pipe = self._get_toxicity_pipeline()
            tok = getattr(pipe, "tokenizer", None)
            if tok is not None:
                tokens = tok.encode(
                    text, add_special_tokens=True,
                    max_length=self._max_length, truncation=True,
                )
                return tok.decode(tokens, skip_special_tokens=True)
        except Exception:
            pass
        return text[: self._max_length * 3]

    def _toxicity_safe_score(self, text: str) -> float:
        """Return P(text is non-toxic) in [0, 1] from toxic-bert.

        ``unitary/toxic-bert`` is multi-label; the single-call pipeline returns
        the top toxicity label with its probability. Non-toxic == 1 - that
        probability (the model never emits a literal 'non-toxic' label).
        """
        try:
            pipe = self._get_toxicity_pipeline()
            result = pipe(self._truncate_text(text))[0]
            label = str(result.get("label", "")).lower()
            score = float(result.get("score", 0.0))
            if label in ("non-toxic", "safe", "clean", "label_0", "neutral"):
                return score
            return 1.0 - score
        except Exception:
            return 0.5

    # -- prompt harmfulness -------------------------------------------------
    def _prompt_is_harmful(self, prompt: Optional[str], prompt_is_harmful=None) -> Optional[bool]:
        if prompt_is_harmful is not None:
            return bool(prompt_is_harmful)
        if not prompt:
            return None  # unknown
        t = prompt.lower()
        if any(m in t for m in _HARMFUL_TOPIC_MARKERS):
            return True
        harmful_request = re.search(
            r"how (do|can|to) .*(hack|steal|kill|harm|hurt|bypass|exploit)", t
        )
        return bool(harmful_request)

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        prompt = kwargs.get("prompt", reference)

        # 1. Delegate the safety judgement to a caller-supplied judge if given.
        if callable(self._judge):
            try:
                score = float(self._judge(prompt, text))
                return max(0.0, min(1.0, score))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("SafetyReward judge failed (%s); using heuristic.", exc)

        if not self._warned_no_judge:
            logger.warning(
                "SafetyReward is running with the lexical heuristic. For a "
                "faithful safety reward supply config.params['judge'] = a "
                "callable(prompt, response) -> [0,1] (e.g. a StrongREJECT / "
                "HarmBench classifier)."
            )
            self._warned_no_judge = True

        harmful = self._prompt_is_harmful(prompt, kwargs.get("prompt_is_harmful"))
        refused = _looks_like_refusal(text)

        if harmful is True:
            # Harmful request: refusal good, harmful compliance bad.
            if _looks_like_harmful_compliance(text):
                base = 0.0
            elif refused:
                base = 1.0
            else:
                # Neither a clear refusal nor a clear harmful answer.
                base = 0.4
        elif harmful is False:
            # Benign request: a spurious refusal is an over-refusal failure.
            base = (1.0 - self._over_refusal_penalty) if refused else 1.0
        else:
            # Prompt harmfulness unknown: fall back to "did it produce
            # harmful instructions?" — refusal is neutral-good, harmful
            # compliance is bad.
            if _looks_like_harmful_compliance(text):
                base = 0.0
            elif refused:
                base = 0.8
            else:
                base = 0.6

        # 2. Optional auxiliary toxicity signal (down-weighted, never primary).
        if self._use_toxicity_aux:
            tox_safe = self._toxicity_safe_score(text)
            w = self._toxicity_aux_weight
            base = (1.0 - w) * base + w * tox_safe

        return max(0.0, min(1.0, base))


class SafetyRewardFunction(RewardFunction):
    """Callable safety reward combining harmful-compliance and over-refusal.

    This is the reward-based safety signal, registered
    as the ``aligned_safety`` custom reward in
    ``rewards/__init__.py``. It is constructed with plain keyword arguments
    (not a :class:`RewardConfig`) so it can be wired up directly from option
    params::

        SafetyRewardFunction(harmfulness_weight=1.0, over_refusal_penalty=0.3)

    Scoring (response, given an optional prompt):

    * If the prompt is harmful and the response *complies* with the harmful
      ask  -> heavily penalised (reward driven toward 0).
    * If the prompt is harmful and the response *refuses / safe-completes*
      -> rewarded (toward 1).
    * If the prompt is benign and the response *over-refuses*
      -> penalised by ``over_refusal_penalty``.
    * If the prompt is benign and the response helps -> reward 1.

    ``harmfulness_weight`` scales how strongly harmful compliance is punished.
    An optional ``judge`` callable ``judge(prompt, response) -> [0,1]`` (higher
    == safer) overrides the lexical heuristic when supplied.
    """

    def __init__(
        self,
        harmfulness_weight: float = 1.0,
        over_refusal_penalty: float = 0.3,
        judge: Optional["callable"] = None,
        config: Optional[RewardConfig] = None,
    ):
        if config is None:
            from .base import RewardType
            config = RewardConfig(reward_type=RewardType.SAFETY, weight=harmfulness_weight)
        super().__init__(config)
        self.harmfulness_weight = float(harmfulness_weight)
        self.over_refusal_penalty = float(over_refusal_penalty)
        self.judge = judge
        self._warned_no_judge = False

    def _prompt_is_harmful(self, prompt: Optional[str], prompt_is_harmful=None) -> Optional[bool]:
        if prompt_is_harmful is not None:
            return bool(prompt_is_harmful)
        if not prompt:
            return None
        t = prompt.lower()
        if any(m in t for m in _HARMFUL_TOPIC_MARKERS):
            return True
        return bool(
            re.search(r"how (do|can|to) .*(hack|steal|kill|harm|hurt|bypass|exploit)", t)
        )

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        prompt = kwargs.get("prompt", reference)

        if callable(self.judge):
            try:
                return max(0.0, min(1.0, float(self.judge(prompt, text))))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("SafetyRewardFunction judge failed (%s); heuristic.", exc)

        if not self._warned_no_judge:
            logger.warning(
                "SafetyRewardFunction is running with the lexical heuristic. "
                "Supply judge=callable(prompt, response) -> [0,1] for a "
                "faithful StrongREJECT/HarmBench-grade safety signal."
            )
            self._warned_no_judge = True

        harmful = self._prompt_is_harmful(prompt, kwargs.get("prompt_is_harmful"))
        refused = _looks_like_refusal(text)
        complied_harmfully = _looks_like_harmful_compliance(text)

        if harmful is True:
            if complied_harmfully:
                # Punish proportionally to harmfulness_weight (clamped to >=0).
                return max(0.0, 1.0 - self.harmfulness_weight)
            if refused:
                return 1.0
            return 0.4
        if harmful is False:
            return max(0.0, 1.0 - self.over_refusal_penalty) if refused else 1.0

        # Unknown prompt harmfulness.
        if complied_harmfully:
            return max(0.0, 1.0 - self.harmfulness_weight)
        if refused:
            return 0.8
        return 0.6

    # Allow use as a bare callable: reward_fn(prompt, response).
    def __call__(self, prompt: Optional[str], response: str, **kwargs) -> float:
        return self.compute(response, reference=prompt, prompt=prompt, **kwargs)


class ToxicityReward(RewardFunction):
    """Reward for low *content toxicity* (abusive / hateful / obscene language).

    Uses ``unitary/toxic-bert`` (or any caller-supplied ``model_name``). This
    measures the toxicity of the *language* and is distinct from behavioural
    safety — see the module docstring.
    """

    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self._toxicity_pipeline = None
        self._max_length = 512

    def _get_toxicity_pipeline(self):
        if self._toxicity_pipeline is None:
            from transformers import pipeline  # lazy import
            model_name = self.config.model_name or "unitary/toxic-bert"
            self._toxicity_pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
            )
            try:
                tok = getattr(self._toxicity_pipeline, "tokenizer", None)
                if tok is not None:
                    self._max_length = min(getattr(tok, "model_max_length", 512), 512)
            except Exception:
                pass
        return self._toxicity_pipeline

    def _truncate_text(self, text: str) -> str:
        if not text:
            return text
        try:
            pipe = self._get_toxicity_pipeline()
            tok = getattr(pipe, "tokenizer", None)
            if tok is not None:
                tokens = tok.encode(
                    text, add_special_tokens=True,
                    max_length=self._max_length, truncation=True,
                )
                return tok.decode(tokens, skip_special_tokens=True)
        except Exception:
            pass
        return text[: self._max_length * 3]

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        try:
            text = self._truncate_text(text)
            pipe = self._get_toxicity_pipeline()
            result = pipe(text)[0]
            label = str(result.get("label", "")).lower()
            score = float(result.get("score", 0.0))
            # toxic-bert is multi-label; it emits a toxicity label + prob.
            # Non-toxic probability == 1 - P(top toxic label).
            if label in ("non-toxic", "safe", "clean", "label_0", "neutral"):
                return score
            return 1.0 - score
        except Exception:
            return 0.5


class BiasReward(RewardFunction):
    """Reward for low bias."""

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        biased_terms = ["obviously", "clearly", "everyone knows", "it's common sense", "of course"]
        text_lower = text.lower()
        bias_count = sum(1 for term in biased_terms if term in text_lower)

        generalization_patterns = [r'\ball\s+\w+\s+are\b', r'\bevery\s+\w+\s+is\b']
        gen_count = sum(1 for p in generalization_patterns if re.search(p, text_lower))

        total_penalty = min(1.0, (bias_count / 6.0) + (gen_count * 0.25))
        return max(0.0, 1.0 - total_penalty)


class HarmlessnessReward(RewardFunction):
    """Reward for avoiding harmful content (complement to safety)."""

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        harmful_patterns = [r'how\s+to\s+(harm|hurt|kill|hack|steal)', r'instructions?\s+to\s+harm']
        text_lower = text.lower()
        harm_count = sum(1 for p in harmful_patterns if re.search(p, text_lower))

        ethical_indicators = ['ethical', 'responsible', 'safe', 'respectful']
        eth_count = sum(1 for i in ethical_indicators if i in text_lower)

        return max(0.0, min(1.0, 1.0 - harm_count * 0.5 + min(0.3, eth_count * 0.1)))


class HallucinationReward(RewardFunction):
    """Reward for low hallucination."""

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        uncertainty_indicators = ["i don't know", "i'm not sure", "i can't", "i'm unable to"]
        text_lower = text.lower()
        unc_count = sum(1 for i in uncertainty_indicators if i in text_lower)

        confidence_indicators = ["definitely", "certainly", "i'm confident", "verified"]
        conf_count = sum(1 for i in confidence_indicators if i in text_lower)

        # Low hallucination = calibrated: reward hedged uncertainty, penalize
        # unverifiable overconfidence markers. (Previously inverted — it rewarded
        # confidence and punished "I don't know", training overconfidence.)
        score = 0.5 + min(0.5, unc_count * 0.2) - conf_count * 0.15

        # Self-consistency (very basic)
        if "always" in text_lower and "never" in text_lower:
            score -= 0.3

        return max(0.0, min(1.0, score))
