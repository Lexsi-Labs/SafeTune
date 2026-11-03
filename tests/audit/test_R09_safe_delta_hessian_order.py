"""Audit R09 (review of #109): with a Hessian Safe Delta keeps upstream's plain argsort order.

Importance is 1/diag(Hinv)^2 per column, so entries of a column tie even with a Hessian. The
data-free tie-break (smaller |delta| first) must not apply there: upstream uses
torch.argsort(importance) (llama2/safedelta/safedelta_runner.py:129 @5440b6f).
"""
import inspect

import pytest
import torch

from safetune.interventions.recover.safe_delta import _adjust_delta, _resolve_hinv


def _run(hessian, s):
    torch.manual_seed(0)
    wa = torch.randn(64, 64)
    ws = wa + 0.1 * torch.randn(64, 64)
    out = _adjust_delta(W_align=wa, W_sft=ws, s=s, hessian=hessian, importance_drop=0.15,
                        percdamp=0.01, ref_columns=64, blocksize=64, sub_block_size=4)
    return out, ws, wa


def _diag_h():
    torch.manual_seed(1)
    return torch.diag(0.5 + torch.rand(64) * 3)


def _full_h():
    torch.manual_seed(1)
    a = torch.randn(64, 128)
    return a @ a.T / 128 + torch.eye(64)


def _plain_argsort_mask(hessian, wa, ws, s):
    """Selection from plain torch.argsort(importance), as upstream does it (same inputs as _run)."""
    dinv2 = torch.diag(_resolve_hinv(hessian, 64, wa.device, 0.01)).reshape(1, -1) ** 2
    importance = torch.ones_like(wa) / dinv2
    loss = ((ws - wa) ** 2 / dinv2).flatten()
    order = torch.argsort(importance.flatten())[:-int(0.15 * importance.numel())]
    mask = torch.zeros_like(loss, dtype=torch.bool)
    mask[order] = torch.cumsum(loss[order], dim=0) <= importance.mean() * s
    return mask.reshape(wa.shape)


# Importance ties within a column, so the tie order, and with it the kept count and the edited
# weights, differs by platform. Compare with the order computed here, not with recorded values.
@pytest.mark.parametrize("make_h", [_diag_h, _full_h])
@pytest.mark.parametrize("s", [0.02, 0.05])
def test_hessian_path_keeps_the_plain_argsort_selection(make_h, s):
    out, ws, wa = _run(make_h(), s)
    mask = _plain_argsort_mask(make_h(), wa, ws, s)
    assert mask.any()
    assert torch.equal(out == ws, mask)


def test_data_free_tie_break_still_applies():
    out, ws, _ = _run(None, 5.0)
    assert int((out == ws).sum()) > 0


def test_fidelity_note_does_not_claim_a_single_deviation():
    # The note lists the data-free tie-break too, so the global budget is not "the ONE" deviation.
    assert "The ONE deviation" not in inspect.getmodule(_adjust_delta).__doc__
