"""
Abliteration: Refusal-Direction Ablation as a red-team attack.

Same primitive as ``safetune.steer.refusal_direction``, exposed under the
Verify pillar with the attack-facing API. Use this when you want to
quantify "how easy is it to remove safety guardrails from a model" rather
than "how much can I strengthen them."

Two attack modes:

* ``runtime_ablate``: install forward hooks that project the residual stream
  onto the orthogonal complement of the refusal direction at every decoder
  layer. Reversible (just remove the hooks). Cheap, no checkpoint write.

* ``weight_orthogonalize``: edit the model's output projection matrices so
  the refusal direction cannot be written to the residual stream at all.
  Permanent (until you restore from a snapshot). Equivalent to a one-shot
  jailbreak-by-weight-editing.

Reference paper: Arditi et al., "Refusal in Language Models Is Mediated by
a Single Direction," NeurIPS 2024, arXiv:2406.11717.

Ethics note: this attack is the canonical baseline for "how robust is the
model's refusal." Reviewers expect it as a stress test. Pair it with the
defense in ``safetune.steer.refusal_direction`` so the table tells both
sides of the story.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from safetune.interventions.steer.refusal_direction import (
    RefusalDirectionConfig,
    RefusalDirectionModel,
    extract_refusal_direction,
    orthogonalize_weights,
    restore_weights,
)


class AbliterationAttack:
    """Refusal-direction ablation packaged as a red-team attack.

    Example::

        attack = AbliterationAttack(model, tokenizer)
        attack.fit(harmful_prompts=harm, harmless_prompts=harmless)
        attack.run(mode="runtime_ablate")   # reversible
        # ... generate outputs, score ASR ...
        attack.revert()
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[RefusalDirectionConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or RefusalDirectionConfig()
        self._direction: Optional[torch.Tensor] = None
        self._layer_idx: Optional[int] = None
        self._runtime: Optional[RefusalDirectionModel] = None
        self._weight_snapshots: Optional[Dict[str, torch.Tensor]] = None

    def fit(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
    ) -> Tuple[torch.Tensor, int]:
        """Extract the refusal direction from contrast prompts."""
        direction, layer_idx, _ = extract_refusal_direction(
            self.model,
            self.tokenizer,
            harmful_prompts=harmful_prompts,
            harmless_prompts=harmless_prompts,
            config=self.config,
        )
        self._direction = direction
        self._layer_idx = layer_idx
        return direction, layer_idx

    def run(self, mode: str = "runtime_ablate", strength: float = 1.0) -> None:
        """Apply the attack. ``mode`` is ``runtime_ablate`` or ``weight_orthogonalize``."""
        if self._direction is None:
            raise RuntimeError("Call .fit(...) before .run(...) to extract the direction.")

        if mode == "runtime_ablate":
            self._runtime = RefusalDirectionModel(
                self.model,
                self._direction,
                mode="ablate",
                strength=strength,
            ).install()
        elif mode == "weight_orthogonalize":
            self._weight_snapshots = orthogonalize_weights(self.model, self._direction)
        else:
            raise ValueError(
                f"mode must be 'runtime_ablate' or 'weight_orthogonalize', got {mode!r}"
            )

    def revert(self) -> None:
        """Undo whichever mode was last applied. Idempotent."""
        if self._runtime is not None:
            self._runtime.remove()
            self._runtime = None
        if self._weight_snapshots is not None:
            restore_weights(self.model, self._weight_snapshots)
            self._weight_snapshots = None

    @property
    def direction(self) -> Optional[torch.Tensor]:
        return self._direction

    @property
    def picked_layer(self) -> Optional[int]:
        return self._layer_idx


__all__ = ["AbliterationAttack"]
