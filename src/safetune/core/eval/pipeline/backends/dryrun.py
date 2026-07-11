"""Dry-run backend.

Returns canned, deterministic strings without loading any model. Used by the
test suite and by ``Generator(... dry_run=True)`` so a CI pipeline can verify
the orchestration without GPU.
"""
from __future__ import annotations

from typing import List, Optional

from .base import GenerationConfig, InferenceBackend


class DryRunBackend(InferenceBackend):
    """Echo a stub response. No model load."""

    def __init__(
        self,
        model: str = "dryrun",
        config: Optional[GenerationConfig] = None,
        template: str = "[DRY RUN] {prompt}",
    ) -> None:
        super().__init__(model=model, config=config, chat_template=False)
        self.template = template

    def generate(self, prompts: List[str]) -> List[str]:
        return [self.template.format(prompt=p[:120]) for p in prompts]
