"""Audit H13 / review of #112: `safetune train --base` must not reach DOOR.

DOORTrainer's `base_model_path` is the NPO reference. The official DOOR code loads `ref_model` from
the same path as the trained model (`train.py:15-29`, both from `args.base_model`), i.e. the initial
policy, which in this repo is the aligned model. `--base` is the pre-alignment base, so forwarding it
would make the reference a different model than the one being trained.
"""
import json
import sys

import pytest

from safetune import cli
from safetune.runner import harden

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


class _Stop(Exception):
    pass


def _run(monkeypatch, tmp_path, trainer_cls, algo):
    seen = {}

    class Spy(trainer_cls):
        def __init__(self, model, tok, *, base_model_path=None, **kwargs):
            seen["base_model_path"] = base_model_path
            raise _Stop

    monkeypatch.setattr(harden, "SpyTrainer", Spy, raising=False)
    monkeypatch.setitem(cli.HARDEN_REGISTRY, algo, "SpyTrainer")
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps({"prompt": f"q {i}?", "response": f"a {i} here"}) + "\n"
                            for i in range(4)))
    monkeypatch.setattr(sys, "argv", ["safetune", "train", "--algo", algo, "--model", TINY,
                                      "--train-dataset", str(data), "--output", str(tmp_path / "o"),
                                      "--base", "some/base"])
    with pytest.raises(_Stop):
        cli.main()
    return seen["base_model_path"]


def test_cli_does_not_forward_base_to_door(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, harden.DOORTrainer, "doorspy") is None


def test_cli_still_forwards_base_to_lox_harden(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, harden.LoXHardenTrainer, "loxspy") == "some/base"
