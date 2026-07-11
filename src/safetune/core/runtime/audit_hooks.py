"""Audit hooks for runtime safety decisions with redaction controls."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def redact_text(text: Optional[str]) -> Optional[str]:
    """Simple redaction helper that masks long content bodies.

    ``None`` and empty strings are passed through unchanged so callers can
    redact optional fields without special-casing.
    """
    if not text:
        return text
    if len(text) <= 32:
        return "[REDACTED]"
    return text[:16] + "...[REDACTED]..." + text[-8:]


@dataclass
class SafetyAuditEvent:
    tenant_id: str
    decision: str
    reason: str
    prompt: Optional[str] = None
    response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self, redact: bool = True) -> Dict[str, Any]:
        payload = {
            "tenant_id": self.tenant_id,
            "decision": self.decision,
            "reason": self.reason,
            "prompt": self.prompt,
            "response": self.response,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
        if redact:
            payload["prompt"] = redact_text(payload.get("prompt", ""))
            payload["response"] = redact_text(payload.get("response", ""))
        return payload
