"""The README's Python blocks run as written, in order, on CPU (ticket ST-06).

By default the README's models are swapped for a small cached one so this runs
in CI. ``SAFETUNE_README_AS_WRITTEN=1`` runs the README's own models
(Qwen2.5-0.5B-Instruct and Qwen2.5-0.5B, about 2 GB of downloads).
Blocks that say they need a GPU are skipped.
"""
import os
import re
from pathlib import Path

import pytest

import safetune
import safetune.runner._registry as registry
import safetune.runner.harden._base as harden_base

README = Path(__file__).resolve().parents[2] / "README.md"
SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"
# Longest id first, so "Qwen/Qwen2.5-0.5B" does not match inside the Instruct id.
SWAP = [("Qwen/Qwen2.5-0.5B-Instruct", SMALL), ("Qwen/Qwen2.5-0.5B", SMALL)]


def readme_python_blocks(skip_gpu=True):
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.S)
    return [b for b in blocks if not (skip_gpu and "needs a GPU" in b)]


def test_readme_has_the_pillar_blocks():
    code = "\n".join(readme_python_blocks())
    for call in ("RefusalDirectionTrainer", "ReStaTrainer", "SafeGradTrainer"):
        assert call in code
    # Only the Evaluate block may be skipped (its judge is a gated 7B model).
    gpu_only = [b for b in readme_python_blocks(skip_gpu=False) if "needs a GPU" in b]
    assert len(gpu_only) == 1 and "evaluate(" in gpu_only[0]


@pytest.mark.slow
def test_readme_python_blocks_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(harden_base, "DEFAULT_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setitem(registry.HARDEN_REGISTRY, "mymethod", None)  # the README registers one
    safetune.configure(device="cpu")
    try:
        ns = {}
        for block in readme_python_blocks():
            if os.environ.get("SAFETUNE_README_AS_WRITTEN") != "1":
                for old, new in SWAP:
                    block = block.replace(old, new)
            exec(compile(block, str(README), "exec"), ns)
    finally:
        safetune.configure(device=None)
    assert Path(ns["checkpoint"]).is_dir()
    assert ns["patched"] is ns["drifted"]
