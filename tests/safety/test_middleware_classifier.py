"""Tests for SafetyMiddleware with real classifier_fn in post_check."""
import pytest


def _make_router_and_middleware(disallowed_categories=None, classifier_fn=None):
    from safetune.core.runtime.tenant_policy_router import TenantPolicyRouter
    from safetune.core.runtime.safety_middleware import SafetyMiddleware

    policies = {
        "test_tenant": {
            "disallowed_categories": disallowed_categories or [],
            "disallowed_tools": [],
            "allowed_tools": [],
        }
    }
    router = TenantPolicyRouter(policies=policies, default_tenant="test_tenant")
    middleware = SafetyMiddleware(router=router, classifier_fn=classifier_fn)
    return middleware


# ────────────────────────────────────────────────────────────────────────────
# post_check with classifier_fn
# ────────────────────────────────────────────────────────────────────────────

def test_post_check_blocks_when_classifier_returns_disallowed():
    """post_check should block when classifier labels response in disallowed category."""

    def violence_classifier(text: str):
        return {"category": "violence"} if "violence" in text.lower() else {"category": "safe"}

    middleware = _make_router_and_middleware(
        disallowed_categories=["violence"],
        classifier_fn=violence_classifier,
    )
    decision = middleware.post_check(
        tenant_id="test_tenant",
        prompt="Tell me about safety.",
        response="I will explain violence techniques in detail...",
    )
    assert decision.allow is False
    assert "blocked_response_category" in decision.reason


def test_post_check_allows_safe_response_with_classifier():
    """post_check should allow when classifier returns a non-disallowed category."""

    def safe_classifier(text: str):
        return {"category": "cooking"}

    middleware = _make_router_and_middleware(
        disallowed_categories=["violence"],
        classifier_fn=safe_classifier,
    )
    decision = middleware.post_check(
        tenant_id="test_tenant",
        prompt="Give me a recipe.",
        response="Here is a pasta recipe!",
    )
    assert decision.allow is True
    assert decision.reason == "ok"


def test_post_check_no_classifier_uses_heuristic_and_warns(caplog):
    """Without classifier_fn, post_check falls back to string heuristic and logs a warning."""
    import logging

    middleware = _make_router_and_middleware(
        disallowed_categories=["violence"],
        classifier_fn=None,  # No classifier
    )
    with caplog.at_level(logging.WARNING):
        decision = middleware.post_check(
            tenant_id="test_tenant",
            prompt="Tell me about violence.",
            response="This response contains the word violence somewhere.",
        )
    # Heuristic should still block since "violence" is literally in the response.
    assert decision.allow is False
    # Warning should have been logged.
    assert any("classifier_fn" in r.message for r in caplog.records), (
        "A warning about missing classifier_fn should be logged"
    )


def test_post_check_no_disallowed_categories_always_allows():
    """If policy has no disallowed categories, post_check should always allow."""
    middleware = _make_router_and_middleware(
        disallowed_categories=[],
        classifier_fn=None,
    )
    decision = middleware.post_check(
        tenant_id="test_tenant",
        prompt="Tell me anything.",
        response="Something potentially problematic goes here.",
    )
    assert decision.allow is True


# ────────────────────────────────────────────────────────────────────────────
# pre_check integration
# ────────────────────────────────────────────────────────────────────────────

def test_pre_check_blocks_disallowed_category():
    def hate_classifier(text: str):
        return {"category": "hate_speech"}

    middleware = _make_router_and_middleware(
        disallowed_categories=["hate_speech"],
        classifier_fn=hate_classifier,
    )
    decision = middleware.pre_check("test_tenant", "Some hateful prompt.")
    assert decision.allow is False


def test_pre_check_blocks_disallowed_tool():
    middleware = _make_router_and_middleware()

    # Inject a disallowed tool on the fly via policy override.
    from safetune.core.runtime.tenant_policy_router import TenantPolicyRouter
    from safetune.core.runtime.safety_middleware import SafetyMiddleware

    policies = {
        "t": {
            "disallowed_categories": [],
            "disallowed_tools": ["shell_exec"],
            "allowed_tools": [],
        }
    }
    router = TenantPolicyRouter(policies=policies, default_tenant="t")
    mw = SafetyMiddleware(router=router)
    decision = mw.pre_check("t", "Run this command.", tool_name="shell_exec")
    assert decision.allow is False
    assert "blocked_tool" in decision.reason
