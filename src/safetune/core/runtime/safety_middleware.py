"""Runtime safety middleware (pre-check/post-check enforcement)."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .audit_hooks import SafetyAuditEvent
from .tenant_policy_router import TenantPolicyRouter

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Default keyword-based category detector
# ──────────────────────────────────────────────────────────────────────────────
# This is a deliberately conservative, transparent fallback so that the
# middleware is *functional out of the box* rather than silently inert when no
# ``classifier_fn`` is supplied.  It is NOT a substitute for a real safety
# classifier — callers handling sensitive workloads should pass ``classifier_fn``.
DEFAULT_CATEGORY_KEYWORDS: Dict[str, list] = {
    "violence": ["kill", "murder", "bomb", "weapon", "attack", "shoot", "assault"],
    "self-harm": ["suicide", "self-harm", "self harm", "kill myself", "hurt myself"],
    "hate-speech": ["hate speech", "racial slur", "ethnic cleansing"],
    "illegal-activity": ["how to hack", "steal", "launder money", "counterfeit", "drug trafficking"],
    "sexual-content": ["explicit sexual", "pornographic"],
    "dangerous-advice": ["how to make a bomb", "how to poison", "how to build a weapon"],
    "privacy-violation": ["social security number", "credit card number", "doxx"],
}


def default_category_classifier(
    text: str,
    keyword_map: Optional[Dict[str, list]] = None,
) -> Dict[str, Any]:
    """Lightweight keyword-based category classifier used as a fallback.

    Returns ``{"category": <name>, "matched_keyword": <kw>}`` for the first
    category whose keyword appears in ``text``, else ``{"category": "unknown"}``.
    The category names match :data:`DEFAULT_CATEGORY_KEYWORDS` (and the CoSA
    taxonomy) so they line up with typical ``policy.disallowed_categories``.
    """
    keyword_map = keyword_map or DEFAULT_CATEGORY_KEYWORDS
    lowered = (text or "").lower()
    for category, keywords in keyword_map.items():
        for kw in keywords:
            if kw in lowered:
                return {"category": category, "matched_keyword": kw}
    return {"category": "unknown"}


@dataclass
class MiddlewareDecision:
    allow: bool
    reason: str
    enforced_response: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SafetyMiddleware:
    """Applies per-tenant policy checks before and after model execution.

    **Pre-check** (``pre_check()``):
        Blocks prompts whose category is in ``policy.disallowed_categories``
        or that invoke a disallowed tool.  When ``classifier_fn`` is provided
        it is used to classify the prompt.  Otherwise a built-in keyword
        detector (:func:`default_category_classifier`) is used so the check is
        *functional out of the box* — a warning is logged the first time the
        fallback is exercised against a real policy.

    **Post-check** (``post_check()``):
        Inspects model responses for disallowed content.  When
        ``classifier_fn`` is provided it classifies the *response* and
        compares the returned category.  When no ``classifier_fn`` is
        available it falls back to the same built-in keyword detector and
        logs a warning to encourage supplying a real classifier.

    The keyword fallback is conservative and transparent — it is NOT a
    substitute for a trained safety classifier.  Production deployments
    should pass an explicit ``classifier_fn``.  Set ``use_default_detector``
    to ``False`` to opt out of the fallback entirely (checks then become
    no-ops when no classifier is supplied, but a warning still fires).
    """

    def __init__(
        self,
        router: TenantPolicyRouter,
        classifier_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        audit_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        redact_audit: bool = True,
        use_default_detector: bool = True,
    ):
        self.router = router
        self.classifier_fn = classifier_fn
        self.audit_sink = audit_sink
        self.redact_audit = redact_audit
        self.use_default_detector = use_default_detector

    # ------------------------------------------------------------------
    def _classify(self, text: str, *, stage: str) -> Dict[str, Any]:
        """Classify ``text`` into a category dict.

        Uses ``classifier_fn`` when available; otherwise falls back to the
        built-in keyword detector (unless ``use_default_detector`` is False).
        """
        if self.classifier_fn is not None:
            try:
                return self.classifier_fn(text) or {}
            except Exception as exc:
                logger.warning(
                    "SafetyMiddleware %s: classifier_fn failed — %s", stage, exc
                )
                return {}
        if self.use_default_detector:
            logger.warning(
                "SafetyMiddleware %s: no classifier_fn set — using built-in "
                "keyword detector. Provide classifier_fn for reliable checks.",
                stage,
            )
            return default_category_classifier(text)
        logger.warning(
            "SafetyMiddleware %s: no classifier_fn set and use_default_detector "
            "is False — category check is a no-op for this call.",
            stage,
        )
        return {}

    # ------------------------------------------------------------------
    def pre_check(
        self,
        tenant_id: str,
        prompt: str,
        tool_name: Optional[str] = None,
    ) -> MiddlewareDecision:
        """Check whether a prompt is allowed under the tenant policy."""
        policy = self.router.resolve(tenant_id)
        # Only classify when the policy actually constrains prompt categories;
        # this keeps the fallback-warning quiet for tool-only policies.
        if policy.disallowed_categories:
            label = self._classify(prompt, stage="pre_check")
        else:
            label = {}
        category = str(label.get("category", "unknown"))

        if tool_name and tool_name in set(policy.disallowed_tools):
            decision = MiddlewareDecision(
                allow=False,
                reason=f"blocked_tool:{tool_name}",
                enforced_response="Tool usage is not allowed for this tenant policy.",
                metadata={"tool_name": tool_name},
            )
        elif tool_name and policy.allowed_tools and tool_name not in set(policy.allowed_tools):
            decision = MiddlewareDecision(
                allow=False,
                reason=f"tool_not_allowlisted:{tool_name}",
                enforced_response="Tool usage is not allowlisted for this tenant policy.",
                metadata={"tool_name": tool_name},
            )
        elif category in set(policy.disallowed_categories):
            decision = MiddlewareDecision(
                allow=False,
                reason=f"blocked_prompt_category:{category}",
                enforced_response="I cannot help with that request.",
                metadata={"category": category},
            )
        else:
            decision = MiddlewareDecision(allow=True, reason="ok", metadata={"category": category})

        self._emit_event(tenant_id, decision, prompt=prompt, response=decision.enforced_response)
        return decision

    # ------------------------------------------------------------------
    def post_check(
        self,
        tenant_id: str,
        prompt: str,
        response: str,
    ) -> MiddlewareDecision:
        """Check whether a generated response is allowed under the tenant policy.

        The response is classified via ``classifier_fn`` when supplied, or via
        the built-in keyword detector otherwise, and the returned ``category``
        is compared against ``policy.disallowed_categories``.  As a defence in
        depth, the literal category name appearing verbatim in the response
        also triggers a block when the keyword detector is in use.
        """
        policy = self.router.resolve(tenant_id)

        if not policy.disallowed_categories:
            decision = MiddlewareDecision(allow=True, reason="ok")
            self._emit_event(tenant_id, decision, prompt=prompt, response=response)
            return decision

        label = self._classify(response, stage="post_check")
        category = str(label.get("category", "unknown"))

        if category in set(policy.disallowed_categories):
            decision = MiddlewareDecision(
                allow=False,
                reason=f"blocked_response_category:{category}",
                enforced_response="I cannot provide that information.",
                metadata={k: v for k, v in label.items()},
            )
            self._emit_event(tenant_id, decision, prompt=prompt, response=response)
            return decision

        # Defence in depth: when relying on the keyword fallback, also block if
        # a disallowed category name appears verbatim in the response.
        if self.classifier_fn is None and self.use_default_detector:
            lower = response.lower()
            for banned in policy.disallowed_categories:
                if str(banned).lower() in lower:
                    decision = MiddlewareDecision(
                        allow=False,
                        reason=f"blocked_response_content:{banned}",
                        enforced_response="I cannot provide that information.",
                        metadata={"matched_keyword": str(banned)},
                    )
                    self._emit_event(tenant_id, decision, prompt=prompt, response=response)
                    return decision

        decision = MiddlewareDecision(allow=True, reason="ok", metadata={"category": category})
        self._emit_event(tenant_id, decision, prompt=prompt, response=response)
        return decision

    # ------------------------------------------------------------------
    def _emit_event(
        self,
        tenant_id: str,
        decision: MiddlewareDecision,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
    ) -> None:
        if self.audit_sink is None:
            return
        event = SafetyAuditEvent(
            tenant_id=tenant_id,
            decision="allow" if decision.allow else "block",
            reason=decision.reason,
            prompt=prompt,
            response=response,
            metadata=decision.metadata or {},
        )
        self.audit_sink(event.as_dict(redact=self.redact_audit))
