"""Regression tests for the Recover correctness fixes on
``fix/runner-and-method-correctness``.

Each test below FAILS against the pre-fix source and PASSES against the
fixed source. The bug being guarded is named in each test's docstring/comment.

Targets:
  1. safetune.recover.safe_lora._compute_projection
       — divide by the Frobenius norm ‖V‖_F, not the squared norm.
  2. safetune.recover.nlsr._nlsr_stage3_transplant
       — tau-gate single-element regions like multi-element ones (sim=1.0).
  3. safetune.recover.safety_vector_restore.apply_safety_vector_restore
       — default skip-list (embed_tokens/lm_head/norm) when target_modules=None.

Run with:
    PYTHONPATH=src python -m pytest tests/safety/test_fix_recover.py -v --no-cov
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Safe-LoRA projection: C = V·Vᵀ / ‖V‖_F  (Frobenius norm, NOT squared)
# ---------------------------------------------------------------------------

def test_compute_projection_uses_frobenius_norm_not_squared():
    """REGRESSION (Safe-LoRA, Hsu et al. arXiv:2405.16833): the projection
    matrix is C = V·Vᵀ / ‖V‖_F. The pre-fix bug divided by the *squared*
    Frobenius norm (V**2).sum(), which scales C wrongly.

    ``_compute_projection(a, b, device)`` builds V = _to_2d(a - b) internally,
    so we pass b = zeros and a = V to make V == a exactly. The function adds a
    fixed eps=1e-12 to the denominator to guard div-by-zero.
    """
    from safetune.recover.safe_lora import _compute_projection

    # A non-trivial small matrix so the squared-norm and norm versions differ.
    V = torch.tensor(
        [[1.0, 2.0, 0.0],
         [0.0, 1.0, 3.0]],
        dtype=torch.float32,
    )
    zeros = torch.zeros_like(V)
    device = torch.device("cpu")

    eps = 1e-12
    fro_norm = torch.sqrt(torch.sum(V ** 2))  # ‖V‖_F
    expected = (V @ V.t()) / (fro_norm + eps)

    out = _compute_projection(V, zeros, device)
    assert out is not None, "_compute_projection returned None for a valid V"
    assert torch.allclose(out, expected, atol=1e-6), (
        "projection does not match V·Vᵀ / ‖V‖_F"
    )

    # Guard against regression to the squared-norm bug.
    squared_version = (V @ V.t()) / (V ** 2).sum()
    assert not torch.allclose(out, squared_version, atol=1e-6), (
        "projection matches the buggy squared-norm version V·Vᵀ / (V**2).sum()"
    )


# ---------------------------------------------------------------------------
# 2. NLSR stage-3 transplant: tau-gate single-element regions too
# ---------------------------------------------------------------------------

class _OneParam(nn.Module):
    """Minimal module exposing a single named parameter for the transplant fn."""

    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.w = nn.Parameter(value.clone())


def test_nlsr_single_element_region_is_gated_out_like_multi_element():
    """REGRESSION (NLSR stage-3 tau gate): a 1-element region whose similarity
    to the donor is >= tau must be GATED OUT (skipped), not transplanted
    unconditionally. The pre-fix code only applied the cosine/tau ``continue``
    when numel() >= 2, so single-element regions slipped through and were always
    transplanted. The fix treats a 1-element similarity as 1.0 (>= tau).

    We compare a 1-element region against a 2-element identical region: with a
    finite tau both should be gated out, leaving the param UNCHANGED.
    """
    from safetune.recover.nlsr import _nlsr_stage3_transplant

    tau = 0.5

    # --- 1-element param (the regression case) ---
    orig_single = torch.tensor([3.0])
    m_single = _OneParam(orig_single)
    donor_single = {"w": torch.tensor([99.0])}  # very different value
    _nlsr_stage3_transplant(
        m_single, donor=donor_single, region_mask=None, blend=1.0, tau=tau
    )
    after_single = m_single.w.detach().clone()
    assert torch.equal(after_single, orig_single), (
        "1-element region was transplanted despite tau gating "
        f"(got {after_single.tolist()}, expected {orig_single.tolist()})"
    )

    # --- 2-element identical-direction param (control: also gated out) ---
    orig_pair = torch.tensor([1.0, 2.0])
    m_pair = _OneParam(orig_pair)
    # Same direction as orig (cos == 1.0 >= tau) -> gated out.
    donor_pair = {"w": torch.tensor([10.0, 20.0])}
    _nlsr_stage3_transplant(
        m_pair, donor=donor_pair, region_mask=None, blend=1.0, tau=tau
    )
    after_pair = m_pair.w.detach().clone()
    assert torch.equal(after_pair, orig_pair), (
        "2-element identical-direction region should be gated out (control)"
    )


def test_nlsr_dissimilar_region_is_still_transplanted():
    """Sanity counterpart: a region whose similarity is BELOW tau must still be
    transplanted, so the fix gates only the degenerate/high-similarity case.
    """
    from safetune.recover.nlsr import _nlsr_stage3_transplant

    tau = 0.5
    orig = torch.tensor([1.0, 2.0])
    m = _OneParam(orig)
    # Opposite direction -> cos = -1.0 < tau -> transplanted.
    donor = {"w": torch.tensor([-5.0, -10.0])}
    _nlsr_stage3_transplant(m, donor=donor, region_mask=None, blend=1.0, tau=tau)
    after = m.w.detach().clone()
    assert torch.equal(after, donor["w"]), (
        "dissimilar (cos < tau) region should be transplanted with blend=1.0"
    )


# ---------------------------------------------------------------------------
# 3. safety_vector_restore default skip-list when target_modules is None
# ---------------------------------------------------------------------------

class _NamedModel(nn.Module):
    """Tiny module whose parameter names mimic embed_tokens / lm_head / norm /
    a normal mlp layer, so we can check the default skip-list by name.
    """

    def __init__(self) -> None:
        super().__init__()
        # nn.ModuleDict / nested modules produce dotted names matching the skip
        # substrings: 'model.embed_tokens.weight', 'model.norm.weight',
        # 'model.layers.0.mlp.weight', and 'lm_head.weight'.
        self.model = nn.ModuleDict(
            {
                "embed_tokens": nn.Linear(4, 4, bias=False),
                "norm": nn.Linear(4, 4, bias=False),
                "layers": nn.ModuleList(
                    [nn.ModuleDict({"mlp": nn.Linear(4, 4, bias=False)})]
                ),
            }
        )
        self.lm_head = nn.Linear(4, 4, bias=False)


def test_safety_vector_restore_default_skip_list_protects_embed_head_norm():
    """REGRESSION (safety_vector_restore default skip-list): with
    ``target_modules=None`` the method must leave params named with
    embed_tokens / lm_head / norm UNCHANGED, while editing a normal layer
    param. The pre-fix code had no default skip-list, so a full-rank alpha=1
    edit overwrote embeddings/LM-head/norm and destroyed capability.

    ``apply_safety_vector_restore(model, aligned, alpha=..., rank=..., ...)``
    mutates ``model`` in place toward ``aligned``. We give the aligned reference
    different values everywhere; the skipped params must still equal their
    originals, the normal mlp param must move.
    """
    from safetune.recover.safety_vector_restore import apply_safety_vector_restore

    torch.manual_seed(0)
    model = _NamedModel()
    # aligned: same architecture, different weights everywhere.
    aligned = _NamedModel()
    with torch.no_grad():
        for p in aligned.parameters():
            p.add_(torch.full_like(p, 1.0))  # guarantee a non-zero delta

    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    # rank=None -> full safety vector, alpha=1 -> full rollback toward aligned.
    apply_safety_vector_restore(model, aligned, alpha=1.0, rank=None, target_modules=None)

    after = dict(model.named_parameters())

    skipped_names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    edited_name = "model.layers.0.mlp.weight"

    # Make sure our name assumptions actually exist.
    for nm in skipped_names + [edited_name]:
        assert nm in before, f"expected param name {nm!r} not found; got {list(before)}"

    for nm in skipped_names:
        assert torch.equal(after[nm].detach(), before[nm]), (
            f"{nm} was modified but should be in the default skip-list"
        )

    assert not torch.equal(after[edited_name].detach(), before[edited_name]), (
        f"{edited_name} should have been edited (it is not in the skip-list)"
    )
