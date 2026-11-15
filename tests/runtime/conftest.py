"""Shared fixtures for the runtime tests."""

import os

import pytest

from safetune.rewards import _code_grader


@pytest.fixture
def unconfined(monkeypatch):
    """The harnesses with the isolate() call replaced by a stub that reports no isolation (and the opt-out that lets a
    Linux grader run a candidate without seccomp, which is refused by default)."""
    monkeypatch.setenv("SAFETUNE_SANDBOX_ALLOW_NO_SECCOMP", "1")
    call = 'state = isolate(os.getcwd(), job.get("require", False))'
    for name in ("_HARNESS", "_HARNESS_MANY"):
        harness = getattr(_code_grader, name)
        assert call in harness
        monkeypatch.setattr(_code_grader, name, harness.replace(call, 'state = {"seccomp": False, "landlock": 0}'))


# With SAFETUNE_EXPECT_ISOLATION=1 a skipped test in the confinement module is a failure: a silent skip is how a
# broken confinement would go unnoticed on a host that was supposed to have it.
_ALLOWED_SKIPS = ("no kernel headers installed", "could not import", "not available in this Python build")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if (
        report.skipped
        and os.environ.get("SAFETUNE_EXPECT_ISOLATION", "").strip().lower() in ("1", "true", "yes")
        and item.module.__name__.endswith("test_code_confinement")
    ):
        reason = str(report.longrepr[2]) if isinstance(report.longrepr, tuple) else str(report.longrepr)
        if not any(allowed in reason for allowed in _ALLOWED_SKIPS):
            report.outcome = "failed"
            report.longrepr = "skipped under SAFETUNE_EXPECT_ISOLATION=1: " + reason
