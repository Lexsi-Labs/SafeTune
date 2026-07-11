"""Shared contract for post-finetune safety patches."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class PatchState:
    """Serialized patch state for apply/revert and artifactization."""
    patch_id: str
    version: str = "v1"
    metadata: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatchVerificationResult:
    """Verification summary after applying a patch."""
    passed: bool
    safety_delta: float = 0.0
    utility_delta: float = 0.0
    reasons: Dict[str, Any] = field(default_factory=dict)


class SafetyPatch:
    """Base contract used by all safety recovery patches.

    Two application modes:

    1. **Dict mode** (``apply(model_state)``) - operates on a plain Python dict.
       Used in unit tests and lightweight CLI workflows.

    2. **PyTorch mode** (``apply_to_model(model)``) - operates in-place on an
       ``nn.Module``. Requires torch. Subclasses should override this method.
    """

    patch_id: str = "base_patch"

    def __init__(self, **params: Any):
        self.params = params
        self._last_state: Optional[PatchState] = None
        # Snapshot of original param tensors for PyTorch-mode revert.
        self._model_param_backup: Optional[Dict[str, Any]] = None

    # ── Dict-based interface ──────────────────────────────────────────────────

    def apply(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def revert(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        if self._last_state is None:
            return model_state
        restored = dict(model_state)
        for key, value in self._last_state.payload.get("original_values", {}).items():
            restored[key] = value
        return restored

    # ── PyTorch model interface ───────────────────────────────────────────────

    def apply_to_model(self, model: Any) -> None:
        """Apply patch in-place to an ``nn.Module``.

        Override in subclasses to implement real weight manipulation.
        Call ``_backup_params(model)`` before modifying weights to enable
        ``revert_model()``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement apply_to_model(). "
            "Override this method to support real PyTorch model patching."
        )

    def revert_model(self, model: Any) -> None:
        """Restore model weights from backup taken before apply_to_model()."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for revert_model().")
        if self._model_param_backup is None:
            raise RuntimeError(
                "No parameter backup found. Ensure apply_to_model() called "
                "_backup_params() before modifying weights."
            )
        with torch.no_grad():  # type: ignore[union-attr]
            for name, param in model.named_parameters():
                if name in self._model_param_backup:
                    param.data.copy_(self._model_param_backup[name])

    def _backup_params(self, model: Any) -> None:
        """Snapshot all named parameter tensors for later revert."""
        if not TORCH_AVAILABLE:
            return
        self._model_param_backup = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "params": dict(self.params),
            "seed": self.params.get("seed"),
            "source_sha": self.params.get("source_sha"),
            "hyperparams": self.params.get("hyperparams", {}),
            "expected_deltas": self.params.get("expected_deltas", {}),
        }

    def serialize(self) -> PatchState:
        return self._last_state or PatchState(patch_id=self.patch_id, metadata=self.metadata())

    def verify(
        self,
        before_metrics: Optional[Dict[str, float]] = None,
        after_metrics: Optional[Dict[str, float]] = None,
        min_safety_improvement: float = 0.0,
        max_utility_regression: float = 0.05,
    ) -> PatchVerificationResult:
        """
        Verify patch outcome using pre/post metrics.
        safety_delta is interpreted as decrease in harmfulness/jailbreak risk.
        utility_delta is interpreted as regression magnitude.
        """
        before = before_metrics or {}
        after = after_metrics or {}
        before_risk = float(before.get("harmfulness_rate", 0.0)) + float(before.get("jailbreak_success_rate", 0.0))
        after_risk = float(after.get("harmfulness_rate", 0.0)) + float(after.get("jailbreak_success_rate", 0.0))
        safety_delta = before_risk - after_risk

        before_util = float(before.get("utility_score", 1.0))
        after_util = float(after.get("utility_score", before_util))
        utility_delta = before_util - after_util

        passed = True
        reasons: Dict[str, Any] = {}
        if safety_delta < float(min_safety_improvement):
            passed = False
            reasons["safety"] = f"safety_delta {safety_delta:.4f} < {min_safety_improvement:.4f}"
        if utility_delta > float(max_utility_regression):
            passed = False
            reasons["utility"] = f"utility_regression {utility_delta:.4f} > {max_utility_regression:.4f}"
        return PatchVerificationResult(
            passed=passed,
            safety_delta=safety_delta,
            utility_delta=utility_delta,
            reasons=reasons,
        )
