"""Gating for the end-to-end integration tests.

There are two ways to run these:

1. **Real checkpoints** (full drift -> recover -> benchmark-eval numbers).
   Every test that runs a trainer on a *drifted* checkpoint and then runs the
   full 7-bench safety + lm-eval utility suite needs real models, which do NOT
   ship with the repo. Provide them via env vars:

       SAFETUNE_DRIFT_MODEL=<id> [SAFETUNE_BASE_MODEL=<id> SAFETUNE_ALIGNED_MODEL=<id>] \\
           pytest tests/integration -m integration

   Tests needing these are skipped unless SAFETUNE_DRIFT_MODEL is set.

2. **Proxy mode** (lightweight end-to-end smoke on a small public model).
   For CI / local dev without the big checkpoints:

       SAFETUNE_PROXY=1 CUDA_VISIBLE_DEVICES=1 pytest tests/integration -m proxy

   This runs each pillar's train/apply/calibrate/unlearn on
   ``Qwen/Qwen2.5-0.5B-Instruct`` (+ small distinct base/aligned refs) followed
   by a cheap generation eval, proving the whole pipeline executes. It does NOT
   reproduce the safety-drift numbers — that needs real checkpoints (mode 1).
   ``proxy``-marked tests only run when SAFETUNE_PROXY is set.

Every test under tests/integration/ is also marked ``integration`` so the
default suite can collect (and skip) them without failing.
"""
import os

import pytest

_REQUIRED = "SAFETUNE_DRIFT_MODEL"
_THIS_DIR = os.path.dirname(__file__)

# Small, public, cache-friendly defaults used in proxy mode. Distinct base /
# aligned refs so recover's weight arithmetic is a real (non-identity) op.
_PROXY_DEFAULTS = {
    "SAFETUNE_DRIFT_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
    "SAFETUNE_BASE_MODEL": "Qwen/Qwen2-0.5B",
    "SAFETUNE_ALIGNED_MODEL": "Qwen/Qwen2-0.5B-Instruct",
}


def _truthy(v):
    return str(v).strip().lower() not in ("", "0", "false", "no")


def _proxy_enabled():
    return _truthy(os.environ.get("SAFETUNE_PROXY", ""))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: end-to-end integration test")
    config.addinivalue_line("markers", "proxy: lightweight small-model proxy test")
    # In proxy mode, fill in small-model defaults for any checkpoint var the
    # user didn't set, so the proxy tests have models to load.
    if _proxy_enabled():
        for k, v in _PROXY_DEFAULTS.items():
            os.environ.setdefault(k, v)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


def pytest_collection_modifyitems(config, items):
    # This hook receives the WHOLE session's items — only touch tests under
    # tests/integration/, never the rest of the suite.
    proxy = _proxy_enabled()
    # Proxy defaults are injected into the env in pytest_configure, so only count
    # a *user-provided* checkpoint as "real".
    have_ckpt = bool(os.environ.get(_REQUIRED)) and not proxy

    skip_no_ckpt = pytest.mark.skip(
        reason=f"needs a real drifted checkpoint via {_REQUIRED} "
               f"(+ SAFETUNE_BASE_MODEL / SAFETUNE_ALIGNED_MODEL for recover), "
               f"or run the lightweight path with SAFETUNE_PROXY=1 -m proxy"
    )
    skip_no_proxy = pytest.mark.skip(
        reason="proxy test — set SAFETUNE_PROXY=1 to run the lightweight "
               "small-model end-to-end smoke"
    )

    for item in items:
        if not str(item.fspath).startswith(_THIS_DIR):
            continue
        item.add_marker(pytest.mark.integration)
        is_proxy = item.get_closest_marker("proxy") is not None
        if is_proxy:
            # Proxy tests run only in proxy mode.
            if not proxy:
                item.add_marker(skip_no_proxy)
        else:
            # Full-eval tests need real checkpoints; never run under proxy-only.
            if not have_ckpt:
                item.add_marker(skip_no_ckpt)
