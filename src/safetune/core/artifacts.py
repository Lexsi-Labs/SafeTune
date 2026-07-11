"""Versioned artifact manager for SafeTune rollback semantics."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from safetune.core.eval.metrics.safety import JudgeBackendOutput, compute_artifact_promotion_gate


@dataclass
class SafetyArtifactBundle:
    """Atomic versioned bundle for adapters/patches/config references."""
    bundle_id: str
    created_at: str
    adapters: Dict[str, str] = field(default_factory=dict)
    patches: Dict[str, str] = field(default_factory=dict)
    config_refs: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "adapters": self.adapters,
            "patches": self.patches,
            "config_refs": self.config_refs,
            "metadata": self.metadata,
        }


class SafetyArtifactManager:
    """Local filesystem artifact manager with rollback-by-switch behavior."""

    def __init__(self, root_dir: str = "./output/safety_artifacts"):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "active_bundle.json"

    def save_bundle(self, bundle: SafetyArtifactBundle) -> Path:
        out = self.root / f"{bundle.bundle_id}.json"
        out.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
        return out

    def create_bundle(
        self,
        adapters: Optional[Dict[str, str]] = None,
        patches: Optional[Dict[str, str]] = None,
        config_refs: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SafetyArtifactBundle:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bundle = SafetyArtifactBundle(
            bundle_id=f"safety_bundle_{ts}",
            created_at=ts,
            adapters=adapters or {},
            patches=patches or {},
            config_refs=config_refs or {},
            metadata=metadata or {},
        )
        self.save_bundle(bundle)
        return bundle

    def activate(self, bundle_id: str) -> None:
        self.state_path.write_text(json.dumps({"active_bundle": bundle_id}, indent=2), encoding="utf-8")

    def get_active_bundle_id(self) -> Optional[str]:
        if not self.state_path.exists():
            return None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return payload.get("active_bundle")
        except Exception:
            return None

    def load_bundle(self, bundle_id: str) -> Optional[SafetyArtifactBundle]:
        path = self.root / f"{bundle_id}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SafetyArtifactBundle(
            bundle_id=payload["bundle_id"],
            created_at=payload["created_at"],
            adapters=payload.get("adapters", {}),
            patches=payload.get("patches", {}),
            config_refs=payload.get("config_refs", {}),
            metadata=payload.get("metadata", {}),
        )

    def rollback(self, target_bundle_id: str) -> Optional[SafetyArtifactBundle]:
        bundle = self.load_bundle(target_bundle_id)
        if bundle is None:
            return None
        self.activate(target_bundle_id)
        return bundle

    def promotion_decision(
        self,
        metrics: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None,
        backend_output: Optional[JudgeBackendOutput] = None,
        tenant_id: Optional[str] = None,
        per_tenant_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate if a bundle should be promoted to active."""
        return compute_artifact_promotion_gate(
            metrics=metrics,
            thresholds=thresholds,
            backend_output=backend_output,
            tenant_id=tenant_id,
            per_tenant_thresholds=per_tenant_thresholds,
        )

    def promote_if_eligible(
        self,
        bundle_id: str,
        metrics: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None,
        backend_output: Optional[JudgeBackendOutput] = None,
        tenant_id: Optional[str] = None,
        per_tenant_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Promote bundle only if gate checks and backend health pass.
        Returns decision payload with promotion outcome.
        """
        decision = self.promotion_decision(
            metrics=metrics,
            thresholds=thresholds,
            backend_output=backend_output,
            tenant_id=tenant_id,
            per_tenant_thresholds=per_tenant_thresholds,
        )
        decision["bundle_id"] = bundle_id
        if decision.get("promotable"):
            self.activate(bundle_id)
            decision["promoted"] = True
        else:
            decision["promoted"] = False
        return decision
