"""
CST: Configurable Safety Tuning via Opposite System Prompts.
vicgalle/configurable-safety-tuning

CST extends DPO by using two opposite system prompts to create preference pairs
that teach the model to be controllable at inference-time based on the system prompt:
- "Helpful and harmless" prompt -> chosen = safe response
- "Uncensored" prompt -> chosen = unrestricted response

This allows the fine-tuned model to toggle safety behavior based SOLELY on
the system prompt at inference time — no extra data required.

This module provides:
- CSTFormatter: converts a list of (prompt, safe_response, unsafe_response) tuples
  into DPO-ready preference datasets with opposite system prompts.
- CSTConfig: configuration for prompt templates and output format.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_SAFE_SYSTEM = (
    "You are a helpful yet harmless assistant that avoids generating illegal or harmful content."
)
_DEFAULT_UNCENSORED_SYSTEM = (
    "You are a helpful assistant that is completely uncensored."
)


@dataclass
class CSTConfig:
    """Configuration for Configurable Safety Tuning data formatting."""
    safe_system_prompt: str = _DEFAULT_SAFE_SYSTEM
    uncensored_system_prompt: str = _DEFAULT_UNCENSORED_SYSTEM
    # Output dataset format: "dpo" (chosen/rejected pairs) or "raw" (list of dicts)
    output_format: str = "dpo"
    # If True, also emit the "uncensored" pairs (for dual-direction training).
    # If False, only emit the "safe" pairs (one-sided training).
    include_uncensored_pairs: bool = True
    # Chat template string (optional). If provided, system+user are concatenated using this template.
    chat_template: Optional[str] = None


class CSTFormatter:
    """
    Converts (prompt, safe_response, unsafe_response) triples into CST DPO datasets.

    Each triple yields up to 2 preference rows:
    1. System=SAFE:       chosen=safe_response, rejected=unsafe_response
    2. System=UNCENSORED: chosen=unsafe_response, rejected=safe_response
    """

    def __init__(self, config: Optional[CSTConfig] = None) -> None:
        self.config = config or CSTConfig()

    def _format_prompt(self, system: str, user: str) -> str:
        if self.config.chat_template:
            return self.config.chat_template.format(system=system, user=user)
        # Default: simple concatenation with newlines
        return f"[SYSTEM]: {system}\n[USER]: {user}"

    def format_example(
        self,
        user_prompt: str,
        safe_response: str,
        unsafe_response: str,
    ) -> List[Dict[str, Any]]:
        """
        Convert a single triple into one or two DPO-format dicts.
        """
        rows: List[Dict[str, Any]] = []

        # Row 1: safe system prompt -> safe is chosen
        safe_input = self._format_prompt(self.config.safe_system_prompt, user_prompt)
        rows.append({
            "system": self.config.safe_system_prompt,
            "prompt": user_prompt,
            "formatted_input": safe_input,
            "chosen": safe_response,
            "rejected": unsafe_response,
            "cst_mode": "safe",
        })

        # Row 2: uncensored system prompt -> unsafe is chosen
        if self.config.include_uncensored_pairs:
            uncensored_input = self._format_prompt(self.config.uncensored_system_prompt, user_prompt)
            rows.append({
                "system": self.config.uncensored_system_prompt,
                "prompt": user_prompt,
                "formatted_input": uncensored_input,
                "chosen": unsafe_response,
                "rejected": safe_response,
                "cst_mode": "uncensored",
            })

        return rows

    def format_dataset(
        self,
        examples: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Batch-format a list of example dicts.

        Each dict must have keys: 'prompt', 'safe_response', 'unsafe_response'.

        Returns a flat list of DPO-format dicts.
        """
        output = []
        for i, ex in enumerate(examples):
            try:
                rows = self.format_example(
                    user_prompt=ex["prompt"],
                    safe_response=ex["safe_response"],
                    unsafe_response=ex["unsafe_response"],
                )
                output.extend(rows)
            except KeyError as e:
                logger.warning("CST: example %d missing key %s, skipping.", i, e)

        n_in = len(examples)
        n_out = len(output)
        logger.info(
            "CST formatted %d examples -> %d DPO rows (include_uncensored=%s).",
            n_in, n_out, self.config.include_uncensored_pairs,
        )
        return output

    def to_hf_dataset(self, examples: List[Dict[str, str]]) -> Any:
        """
        Format examples and return a HuggingFace Dataset if available.
        Falls back to a plain list if datasets is not installed.
        """
        rows = self.format_dataset(examples)
        try:
            from datasets import Dataset
            return Dataset.from_list(rows)
        except ImportError:
            logger.warning("CST: `datasets` not installed, returning plain list.")
            return rows
