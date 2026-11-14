"""Shared fixtures for the runtime tests."""

import pytest

from safetune.rewards import _code_grader


@pytest.fixture
def unconfined(monkeypatch):
    """The harnesses with the isolate() call replaced by a stub that reports no isolation."""
    call = 'state = isolate(os.getcwd(), job.get("require", False))'
    for name in ("_HARNESS", "_HARNESS_MANY"):
        harness = getattr(_code_grader, name)
        assert call in harness
        monkeypatch.setattr(_code_grader, name, harness.replace(call, 'state = {"seccomp": False, "landlock": 0}'))
