"""
DeRTa: Decoupled Refusal Training.
RobustNLP/DeRTa — ACL 2025

Augments safety training data so that models learn to refuse at *any position*
in a response, not just at the beginning. Two key techniques:
1. MLE with Harmful Response Prefix: prepends varying-length harmful prefixes
   before the safe refusal response. The prefix is context, not a target: rows
   carry it as ``prefix_text`` and the trainer masks it, as the authors do
   (their prefix rows put the harmful words in the masked ``prefix`` field).
2. Reinforced Transition Optimization (RTO): one row per example whose response
   is the full harmful response; the trainer relabels every response token to
   the refusal token ("Sorry"), so the model learns to transition to refusal at
   every position (the authors' ``Sorry_data``).

``DeRTaConfig(legacy=True)`` (or ``safetune.configure(legacy_derta=True)``
through the trainers) restores the old rows: RTO rows were harmful prefix + safe
response at several cut points, and the prefix tokens were training targets.
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# RTO trains the model to emit this token at every position of a harmful
# response. The authors hardcode 19701, which is "Sorry" in the Llama-3
# tokenizer (their RTO rows are named `Sorry_data`); on any other tokenizer
# 19701 is an unrelated token, so derive the id from the text instead.
RTO_REFUSAL_TEXT = "Sorry"


def refusal_token_id(tokenizer: Any, text: str = RTO_REFUSAL_TEXT) -> int:
    """The single token id of ``text`` under ``tokenizer`` (19701 on Llama-3).

    Raises when ``text`` doesn't tokenize to exactly one token under this
    tokenizer -- silently taking the first sub-token of a fragmented word
    (e.g. "S" of "S" + "or" + "ry") would train the model to emit a
    meaningless token instead of a refusal. Pass a ``text`` that tokenizes
    cleanly under this tokenizer, or an explicit ``rto_refusal_token_id``
    to the trainer to skip this derivation entirely.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"tokenizer encodes {text!r} to no tokens")
    if len(ids) != 1:
        raise ValueError(
            f"tokenizer encodes {text!r} to {len(ids)} tokens {ids}, not one "
            f"({[tokenizer.decode([i]) for i in ids]!r}); pass a text that "
            f"tokenizes to a single token under this tokenizer, or an "
            f"explicit rto_refusal_token_id"
        )
    return int(ids[0])


@dataclass
class DeRTaConfig:
    """Configuration for DeRTa data augmentation."""
    # Number of prefix-length variants to generate per example
    num_prefix_variants: int = 5
    # Maximum fraction of the harmful response to use as prefix
    max_prefix_ratio: float = 0.8
    # Minimum prefix length in tokens/words
    min_prefix_length: int = 3
    # Whether to also generate RTO transition pairs
    enable_rto: bool = True
    # Seed for reproducibility
    seed: int = 42
    # True: the old RTO rows (harmful prefix + safe response at several cuts)
    legacy: bool = False


class DeRTaFormatter:
    """
    Augments (prompt, harmful_response, safe_response) triples into
    DeRTa-style training data.

    Usage::

        formatter = DeRTaFormatter(config)
        augmented = formatter.augment_dataset(examples)
    """

    def __init__(self, config: Optional[DeRTaConfig] = None) -> None:
        self.config = config or DeRTaConfig()
        self._rng = random.Random(self.config.seed)

    def _split_into_segments(self, text: str) -> List[str]:
        """Split text into word-level segments."""
        return text.split()

    def augment_example(
        self,
        prompt: str,
        harmful_response: str,
        safe_response: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate augmented training examples from a single triple.

        Returns a list of dicts with keys:
        - 'prompt': the original prompt
        - 'response': 'mle_prefix': harmful prefix + safe refusal; 'rto': the
          harmful response
        - 'prefix_text': 'mle_prefix' rows: the harmful prefix (not a target)
        - 'augmentation': type of augmentation ('mle_prefix' or 'rto')
        - 'prefix_ratio': fraction of harmful response used as prefix
        """
        harmful_words = self._split_into_segments(harmful_response)
        results: List[Dict[str, Any]] = []

        if len(harmful_words) < self.config.min_prefix_length:
            # Too short for meaningful prefix augmentation
            results.append({
                "prompt": prompt,
                "response": safe_response,
                "augmentation": "original",
                "prefix_ratio": 0.0,
            })
            return results

        # 1. MLE with Harmful Response Prefix
        max_len = max(
            self.config.min_prefix_length,
            int(len(harmful_words) * self.config.max_prefix_ratio),
        )
        prefix_lengths = set()
        for _ in range(self.config.num_prefix_variants):
            plen = self._rng.randint(self.config.min_prefix_length, max_len)
            prefix_lengths.add(min(plen, len(harmful_words)))

        for plen in sorted(prefix_lengths):
            prefix = " ".join(harmful_words[:plen])
            augmented_response = f"{prefix} {safe_response}"
            results.append({
                "prompt": prompt,
                "response": augmented_response,
                "prefix_text": prefix,
                "augmentation": "mle_prefix",
                "prefix_ratio": plen / len(harmful_words),
            })

        # 2. Reinforced Transition Optimization (RTO): the harmful response,
        # every token of which the trainer relabels as the refusal token.
        if self.config.enable_rto and not self.config.legacy:
            results.append({
                "prompt": prompt,
                "response": harmful_response,
                "augmentation": "rto",
                "prefix_ratio": 1.0,
            })
        elif self.config.enable_rto:
            # Create transition pairs at multiple positions
            step = max(1, len(harmful_words) // self.config.num_prefix_variants)
            for i in range(step, len(harmful_words), step):
                prefix = " ".join(harmful_words[:i])
                # The model should output the safe response after seeing this prefix
                results.append({
                    "prompt": prompt,
                    "response": f"{prefix} {safe_response}",
                    "prefix_text": prefix,
                    "augmentation": "rto",
                    "prefix_ratio": i / len(harmful_words),
                })

        return results

    def augment_dataset(
        self,
        examples: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Batch augment a list of examples.

        Each dict must have: 'prompt', 'harmful_response', 'safe_response'.
        """
        output = []
        for i, ex in enumerate(examples):
            try:
                rows = self.augment_example(
                    prompt=ex["prompt"],
                    harmful_response=ex["harmful_response"],
                    safe_response=ex["safe_response"],
                )
                output.extend(rows)
            except KeyError as e:
                logger.warning("DeRTa: example %d missing key %s, skipping.", i, e)

        logger.info(
            "DeRTa: augmented %d examples -> %d training rows.", len(examples), len(output)
        )
        return output
