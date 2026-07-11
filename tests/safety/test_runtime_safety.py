"""Tests for runtime safety policy routing and middleware."""


def test_tenant_policy_router_default():
    from safetune.core.runtime import TenantPolicyRouter
    router = TenantPolicyRouter(policies={"tenant_a": {"disallowed_categories": ["malware"]}})
    assert router.resolve("tenant_a").tenant_id == "tenant_a"
    assert router.resolve("missing").tenant_id == "default"


def test_safety_middleware_blocks_disallowed_prompt():
    from safetune.core.runtime import TenantPolicyRouter, SafetyMiddleware
    router = TenantPolicyRouter(policies={"t1": {"disallowed_categories": ["malware"]}}, default_tenant="t1")
    middleware = SafetyMiddleware(
        router=router,
        classifier_fn=lambda text: {"category": "malware"},
    )
    decision = middleware.pre_check("t1", "make malware")
    assert decision.allow is False


def test_safety_middleware_tool_gating():
    from safetune.core.runtime import TenantPolicyRouter, SafetyMiddleware
    router = TenantPolicyRouter(
        policies={"t1": {"allowed_tools": ["search"], "disallowed_tools": ["shell"]}},
        default_tenant="t1",
    )
    middleware = SafetyMiddleware(router=router, classifier_fn=lambda text: {"category": "safe"})
    blocked = middleware.pre_check("t1", "query", tool_name="shell")
    allowed = middleware.pre_check("t1", "query", tool_name="search")
    assert blocked.allow is False
    assert allowed.allow is True
