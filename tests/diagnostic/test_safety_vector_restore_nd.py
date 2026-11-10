"""apply_safety_vector_restore on a weight with more than two dimensions (a convolution kernel).

``rank`` keeps the top singular directions of ``aligned - drifted`` "per weight matrix". For a 3-D
or 4-D weight the SVD ran batched over the last two dimensions, one small SVD per output channel,
while the rank was clamped with the first two. The update applied to a Conv1d kernel with rank=1
had rank 3, not 1. The default selection takes every parameter outside embed_tokens / lm_head /
norm, so it reaches any conv weight a model has. A conv kernel is projected as the matrix
(out_channels, in_channels * kernel).
"""
import pytest
import torch

from safetune.recover import apply_safety_vector_restore


def _pair(make):
    torch.manual_seed(0)
    drifted = make()
    torch.manual_seed(1)
    aligned = make()
    return drifted, aligned


@pytest.mark.parametrize("make", [
    lambda: torch.nn.Sequential(torch.nn.Conv1d(6, 5, 3, bias=False)),
    lambda: torch.nn.Sequential(torch.nn.Conv2d(4, 5, 2, bias=False)),
], ids=["conv1d", "conv2d"])
def test_conv_weight_gets_a_rank_r_update_of_its_flattened_matrix(make):
    drifted, aligned = _pair(make)
    w0 = drifted[0].weight.detach().clone()
    v = (aligned[0].weight.detach() - w0).reshape(w0.shape[0], -1)

    apply_safety_vector_restore(drifted, aligned, rank=1, target_modules=["0.weight"])

    update = (drifted[0].weight.detach() - w0).reshape(w0.shape[0], -1)
    assert torch.linalg.matrix_rank(update, atol=1e-5).item() == 1
    U, S, Vh = torch.linalg.svd(v, full_matrices=False)
    assert torch.allclose(update, (U[:, :1] * S[:1]) @ Vh[:1], atol=1e-5)


def test_linear_weight_is_unchanged_by_the_fix():
    drifted, aligned = _pair(lambda: torch.nn.Sequential(torch.nn.Linear(8, 6, bias=False)))
    w0 = drifted[0].weight.detach().clone()
    v = aligned[0].weight.detach() - w0
    apply_safety_vector_restore(drifted, aligned, rank=2, target_modules=["0.weight"])
    U, S, Vh = torch.linalg.svd(v, full_matrices=False)
    assert torch.allclose(drifted[0].weight.detach() - w0, (U[:, :2] * S[:2]) @ Vh[:2], atol=1e-5)
