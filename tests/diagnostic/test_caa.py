"""Diagnostic: end-to-end test of CAA steering."""
from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn


class _Inner(nn.Module):
    def __init__(self, hidden: int, n_layers: int, vocab: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids.long())
        for layer in self.layers:
            h = torch.nn.functional.gelu(layer(h)) + h
        return self.norm(h)


class _Wrap(nn.Module):
    def __init__(self, hidden: int = 16, n_layers: int = 4, vocab: int = 200) -> None:
        super().__init__()
        self.model = _Inner(hidden, n_layers, vocab)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Any:
        return self.lm_head(self.model(input_ids))


class _Tok:
    pad_token_id = 0

    def __call__(self, batch, return_tensors=None, padding=True, truncation=True, max_length=32):
        ids = [[(ord(c) % 199) + 1 for c in s][:max_length] for s in batch]
        n = max(len(s) for s in ids)
        attn = [[1] * len(s) + [0] * (n - len(s)) for s in ids]
        ids = [s + [self.pad_token_id] * (n - len(s)) for s in ids]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


@pytest.fixture
def fixture():
    torch.manual_seed(0)
    return _Wrap(), _Tok()


def test_caa_extract_returns_per_layer_vectors(fixture):
    model, tok = fixture
    from safetune.steer import extract_caa_vectors, CAAConfig

    cfg = CAAConfig(target_layers=[0, 1, 2, 3], normalize=False)
    vecs = extract_caa_vectors(
        model,
        tok,
        positive_prompts=["good response", "helpful answer", "polite reply", "kind word"],
        negative_prompts=["rude reply", "unhelpful", "dismissive", "harsh tone"],
        config=cfg,
    )
    assert set(vecs.keys()) == {0, 1, 2, 3}
    for v in vecs.values():
        assert v.dim() == 1 and v.shape[0] == 16


def test_caa_steering_shifts_logits(fixture):
    """Applying CAA hooks should change the output logits."""
    model, tok = fixture
    from safetune.steer import extract_caa_vectors, CAAConfig, CAAModel

    inputs = tok(["test prompt please"])["input_ids"]
    with torch.no_grad():
        baseline = model(inputs).clone()

    cfg = CAAConfig(target_layers=[0, 1, 2, 3], normalize=False)
    vecs = extract_caa_vectors(
        model,
        tok,
        positive_prompts=["polite", "kind", "helpful", "honest"],
        negative_prompts=["rude", "harsh", "dismissive", "deceptive"],
        config=cfg,
    )

    with CAAModel(model, vecs, strength=3.0):
        with torch.no_grad():
            steered = model(inputs)

    delta = (steered - baseline).abs().max().item()
    assert delta > 1e-4, f"CAA hooks did not shift logits (max delta = {delta:.2e})"


def test_caa_context_manager_removes_hooks(fixture):
    """After exiting the CAAModel context, the model must produce baseline logits."""
    model, tok = fixture
    from safetune.steer import extract_caa_vectors, CAAConfig, CAAModel

    inputs = tok(["test"])["input_ids"]
    with torch.no_grad():
        before = model(inputs).clone()

    cfg = CAAConfig(target_layers=[0, 1, 2, 3])
    vecs = extract_caa_vectors(
        model, tok,
        positive_prompts=["a", "b", "c", "d"],
        negative_prompts=["e", "f", "g", "h"],
        config=cfg,
    )
    with CAAModel(model, vecs, strength=10.0):
        pass  # immediately exit

    with torch.no_grad():
        after = model(inputs)
    assert torch.allclose(before, after), "CAAModel left hooks installed after context exit"


def test_caa_extract_uses_config_batch_size(fixture):
    model, tok = fixture
    from safetune.steer import CAAConfig, extract_caa_vectors

    calls = []
    handle = model.register_forward_pre_hook(lambda _m, _a: calls.append(1))
    try:
        extract_caa_vectors(
            model,
            tok,
            positive_prompts=["a", "b", "c", "d"],
            negative_prompts=["e", "f", "g", "h"],
            config=CAAConfig(target_layers=[0], batch_size=1),
        )
    finally:
        handle.remove()
    assert len(calls) == 8


# --- batch size: precedence, validation, equivalence, runner reach -------------------------------

_POS = [f"polite reply {i}" for i in range(4)]  # one length for every prompt, so no padding
_NEG = [f"rude replies {i}" for i in range(4)]
assert len({len(p) for p in _POS + _NEG}) == 1


@pytest.fixture
def runtime(monkeypatch):
    """Fresh process-wide runtime settings, restored after the test."""
    import safetune.config as sc

    monkeypatch.setattr(sc, "_RUNTIME", sc.RuntimeConfig())
    return sc


def _count_forwards(model, tok, config, pos=_POS, neg=_NEG):
    from safetune.steer import extract_caa_vectors

    calls = []
    handle = model.register_forward_pre_hook(lambda _m, _a: calls.append(1))
    try:
        vecs = extract_caa_vectors(model, tok, pos, neg, config)
    finally:
        handle.remove()
    return len(calls), vecs


def test_caa_default_batch_size_is_16(fixture, runtime):
    model, tok = fixture
    from safetune.steer import CAAConfig

    assert CAAConfig().batch_size is None  # resolved from the runtime setting, 16 unless configured
    assert runtime.get_config().steer_extract_batch_size == 16
    prompts = [f"prompt number {i:02d}" for i in range(20)]  # 20 per side: 2 batches of <=16 each
    n, _ = _count_forwards(model, tok, CAAConfig(target_layers=[0]), prompts, prompts)
    assert n == 4


def test_caa_batch_size_precedence(fixture, runtime):
    """explicit argument > configure() > default."""
    model, tok = fixture
    from safetune.steer import CAAConfig

    runtime.configure(steer_extract_batch_size=2)
    n, _ = _count_forwards(model, tok, CAAConfig(target_layers=[0]))
    assert n == 4  # 2 batches per side
    n, _ = _count_forwards(model, tok, CAAConfig(target_layers=[0], batch_size=1))
    assert n == 8  # the explicit value wins over configure()
    runtime.configure(steer_extract_batch_size=None)
    n, _ = _count_forwards(model, tok, CAAConfig(target_layers=[0]))
    assert n == 2  # back to 16: one batch per side


def test_caa_batch_size_does_not_change_vectors(fixture):
    model, tok = fixture
    from safetune.steer import CAAConfig

    ref = None
    for bs in (16, 4, 3, 1):
        for pool in ("last_token", "mean"):
            cfg = CAAConfig(target_layers=[0, 2, 3], pool_method=pool, batch_size=bs)
            _, vecs = _count_forwards(model, tok, cfg)
            if bs == 16:
                ref = ref or {}
                ref[pool] = vecs
                continue
            for layer, v in vecs.items():
                assert torch.allclose(v, ref[pool][layer], atol=1e-6), (bs, pool, layer)


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "4"])
def test_caa_rejects_invalid_batch_size(fixture, bad):
    """0 used to raise ZeroDivisionError and a negative size ran no batch and returned {}."""
    model, tok = fixture
    from safetune.steer import CAAConfig, extract_caa_vectors

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        extract_caa_vectors(model, tok, _POS, _NEG, CAAConfig(target_layers=[0], batch_size=bad))


def test_configure_rejects_invalid_extract_batch_size(fixture, runtime):
    model, tok = fixture
    from safetune.steer import CAAConfig, extract_caa_vectors

    runtime.configure(steer_extract_batch_size=0)
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        extract_caa_vectors(model, tok, _POS, _NEG, CAAConfig(target_layers=[0]))


def test_extractor_default_follows_configure(fixture, runtime):
    """Every extractor user (refusal direction, AdaSteer, AlphaSteer, ...) shares this default."""
    model, tok = fixture
    from safetune.core.runtime.inference.vector_extraction import (
        SteeringVectorExtractor,
        VectorExtractionConfig,
    )

    runtime.configure(steer_extract_batch_size=2)
    calls = []
    handle = model.register_forward_pre_hook(lambda _m, _a: calls.append(1))
    try:
        SteeringVectorExtractor(model, tok, VectorExtractionConfig(target_layers=[0])).extract(_POS, _NEG)
    finally:
        handle.remove()
    assert len(calls) == 4


def test_yaml_runtime_block_sets_extract_batch_size(tmp_path, runtime):
    from safetune.config import SafeTuneConfig

    path = tmp_path / "cfg.yaml"
    path.write_text("command: eval\nruntime:\n  steer_extract_batch_size: 3\n")
    runtime.configure(**SafeTuneConfig.from_yaml(str(path)).runtime)
    assert runtime.get_config().steer_extract_batch_size == 3


def test_caa_and_cast_trainers_pass_batch_size(fixture, monkeypatch):
    model, tok = fixture
    import safetune.steer as S
    from safetune.runner.steer import CASTTrainer, CAATrainer

    seen = []

    def fake_extract(_model, _tok, _pos, _neg, cfg):
        seen.append(cfg.batch_size)
        return {}

    monkeypatch.setattr(S, "extract_caa_vectors", fake_extract)
    monkeypatch.setattr(S, "CAAModel", lambda *a, **k: None)
    monkeypatch.setattr(S, "fit_cast_condition", lambda *a, **k: None)
    monkeypatch.setattr(S, "CASTModel", lambda *a, **k: None)
    for cls in (CAATrainer, CASTTrainer):
        for bs in (3, None):
            cls(model, tok, model_id="toy", batch_size=bs).calibrate(harmful=_NEG, harmless=_POS)
    assert seen == [3, None, 3, None]
