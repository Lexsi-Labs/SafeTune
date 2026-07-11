"""
DeRTa: Decoupled Refusal Training.
RobustNLP/DeRTa — ACL 2025

Augments safety training data so that models learn to refuse at *any position*
in a response, not just at the beginning. Two key techniques:
1. MLE with Harmful Response Prefix: prepends varying-length harmful prefixes
   before the safe refusal response.
2. Reinforced Transition Optimization (RTO): creates training pairs where the
   model must transition from harmful content to refusal at every position.
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
        - 'response': the augmented response (harmful prefix + safe refusal)
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
                "augmentation": "mle_prefix",
                "prefix_ratio": plen / len(harmful_words),
            })

        # 2. Reinforced Transition Optimization (RTO)
        if self.config.enable_rto:
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
