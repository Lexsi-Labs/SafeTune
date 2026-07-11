"""Tenant policy routing for runtime safety enforcement."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TenantPolicy:
    tenant_id: str
    allowed_categories: list = field(default_factory=list)
    disallowed_categories: list = field(default_factory=list)
    refusal_style: str = "polite"
    escalation_rules: Dict[str, Any] = field(default_factory=dict)
    gate_thresholds: Dict[str, float] = field(default_factory=dict)
    allowed_tools: list = field(default_factory=list)
    disallowed_tools: list = field(default_factory=list)


class TenantPolicyRouter:
    """Resolves per-tenant policy profiles with a deterministic default."""

    def __init__(self, policies: Optional[Dict[str, Dict[str, Any]]] = None, default_tenant: str = "default"):
        self.default_tenant = default_tenant
        policies = policies or {}
        self._policies: Dict[str, TenantPolicy] = {}
        for tenant_id, payload in policies.items():
            self._policies[tenant_id] = TenantPolicy(tenant_id=tenant_id, **payload)
        if self.default_tenant not in self._policies:
            self._policies[self.default_tenant] = TenantPolicy(tenant_id=self.default_tenant)

    def resolve(self, tenant_id: Optional[str]) -> TenantPolicy:
        if tenant_id and tenant_id in self._policies:
            return self._policies[tenant_id]
        return self._policies[self.default_tenant]
