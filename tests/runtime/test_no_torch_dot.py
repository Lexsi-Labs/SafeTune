"""Dot products avoid torch.dot, which fails on CUDA with torch 2.10.0+cu126.

There ``torch.dot`` (cublasSdot) raises CUBLAS_STATUS_NOT_SUPPORTED even for a 1000-element vector;
torch 2.6 on the same GPU is fine. CPU has no such failure, so these tests make torch.dot raise the
same way and run every code path that used it.
"""
import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn

SRC = Path(__file__).resolve().parents[2] / "src" / "safetune"


@pytest.fixture
def broken_dot(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasSdot`")
    monkeypatch.setattr(torch, "dot", boom)
    monkeypatch.setattr(torch.Tensor, "dot", boom)


def _tiny():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 1))


def _loss(model, batch):
    return model(batch["x"]).pow(2).mean()


def test_safe_delta_projection(broken_dot):
    from safetune.core.optim.safe_delta import SafeDeltaWrapper
    w = SafeDeltaWrapper(_tiny())
    got = w._project_gradient(torch.tensor([1.0, -2.0]), torch.tensor([0.0, 1.0]))
    assert torch.allclose(got, torch.tensor([1.0, 0.0]))


def test_antibody_attenuation(broken_dot):
    from safetune.core.optim.antibody import AntibodyWrapper
    m = _tiny()
    w = AntibodyWrapper(m)
    batch = {"x": torch.randn(2, 4)}
    w.capture_harmful_reference(m, batch, loss_fn=_loss)
    _loss(m, batch).backward()
    assert w.attenuate_gradients(m) == pytest.approx(1.0, abs=1e-5)   # same batch: cosine 1


def test_seal_selector_scores(broken_dot):
    from safetune.core.data_compiler.seal_selector import SEALDataSelector
    m = _tiny()
    anchor = {"x": torch.randn(2, 4)}
    scores = SEALDataSelector(m).score_examples([anchor], _loss, lambda mod: _loss(mod, anchor))
    assert scores[0] == pytest.approx(1.0, abs=1e-5)   # main: the error was caught and scored 0.0


def test_cast_similarity(broken_dot):
    from safetune.interventions.steer.cast import _cast_similarity
    v = torch.tensor([1.0, 0.0, 0.0])
    assert 0.0 < _cast_similarity(torch.tensor([2.0, 1.0, 0.0]), v) <= 1.0


def test_tracin_influence(broken_dot):
    from safetune.interventions.unlearn.influence import tracin_influence
    ex = {"x": torch.randn(2, 4)}
    out = tracin_influence(_tiny(), [ex], ex, _loss)
    assert out[0] > 0   # an example's influence on itself is its squared gradient norm


def test_asft_penalty_and_constraint(broken_dot):
    from safetune.core.optim.asft import AsFTWrapper
    m = _tiny()
    aligned = {k: v.clone() for k, v in m.state_dict().items()}
    base = {k: v - 0.1 for k, v in aligned.items()}
    w = AsFTWrapper(m, aligned, base)
    with torch.no_grad():
        m[0].weight.add_(0.05)
    assert float(w.compute_subspace_penalty()) > 0
    _loss(m, {"x": torch.randn(2, 4)}).backward()
    with w.apply_subspace_constraint():
        pass


def test_no_torch_dot_left_in_src():
    pattern = re.compile(r"\btorch\.dot\(|(?<!np)(?<!numpy)\.dot\(")
    hits = []
    for path in SRC.rglob("*.py"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if pattern.search(code):
                hits.append(f"{path.relative_to(SRC)}:{n}: {line.strip()}")
    assert not hits, "\n".join(hits)
