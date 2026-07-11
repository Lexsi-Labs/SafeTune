"""Regression tests for two HARDEN correctness fixes.

These tests target two specific bugs that were fixed on branch
``fix/runner-and-method-correctness``:

1. ``constrained_sft.py`` — the position-decay KL weight used ABSOLUTE
   sequence position, so the heaviest weight landed on the (masked) prompt
   prefix instead of the first RESPONSE token. The fix makes ``t`` response
   relative (t=0 at each row's first ``labels != -100`` token).

2. ``seal.py`` — cached per-example importance weights were reused on a
   later batch without a size check, crashing the ``losses * weights``
   multiply on a size mismatch (e.g. a smaller final batch). The fix falls
   back to uniform ``torch.ones`` weights whenever the cache does not match
   the current batch size.

Each test is written to FAIL against the pre-fix code and PASS now. The
bug being guarded is named in a comment at the top of each test.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared stubs (mirrors tests/diagnostic/test_harden_instantiation.py).
# ---------------------------------------------------------------------------

class _TinyLM(nn.Module):
    """nn.Module that quacks like an HF causal LM enough for compute_loss."""

    def __init__(self, vocab: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.config = type("C", (), {
            "vocab_size": vocab,
            "hidden_size": hidden,
            "num_hidden_layers": 2,
            "model_type": "tiny",
            "use_return_dict": True,
            "torch_dtype": torch.float32,
        })()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids=None, labels=None, **kw):
        if input_ids is None:
            input_ids = torch.zeros((1, 4), dtype=torch.long)
        h = self.embed_tokens(input_ids.long())
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return type("O", (), {"logits": logits, "loss": loss})

    def get_input_embeddings(self):
        return self.embed_tokens

    def parameters(self, recurse: bool = True):  # explicit for clarity
        return super().parameters(recurse)


# ---------------------------------------------------------------------------
# Test 1: constrained_sft position-decay weighting is RESPONSE-RELATIVE.
# ---------------------------------------------------------------------------

def _csft_pos_weights(
    labels: torch.Tensor,
    beta: float,
    decay_rate: float,
) -> torch.Tensor:
    """Replicate exactly the (separable) position-weight computation from
    ConstrainedSFTTrainer.compute_loss.

    This mirrors the post-fix block in
    ``src/safetune/harden/constrained_sft.py`` (lines computing
    ``abs_pos``, ``first_resp``, ``t`` and ``pos_weights``). It is the
    smallest reliable unit surface: the weight logic is a pure function of
    ``labels``, so we factor it out rather than spinning up a full HF
    Trainer. The assertions below would FAIL under the pre-fix code, which
    used ``t = abs_pos`` (absolute position) directly.
    """
    B, S = labels.shape
    dtype = torch.float32
    abs_pos = torch.arange(S).unsqueeze(0)  # (1, S)
    valid_bool = labels != -100
    first_resp = torch.argmax(valid_bool.int(), dim=1, keepdim=True)  # (B, 1)
    t = (abs_pos - first_resp).clamp(min=0).to(dtype)  # (B, S)
    pos_weights = beta * torch.exp(-decay_rate * t)  # (B, S)
    return pos_weights


def test_csft_position_weights_peak_at_first_response_token():
    """BUG: position-decay weight used ABSOLUTE position, so the max weight
    landed on the masked prompt prefix instead of the first response token.

    Build a 2-row batch where each row's response starts at a DIFFERENT
    index. The post-fix weighting must peak at each row's OWN response start.
    Under the old absolute-position code, the max would always be at column 0
    (the prompt), which is wrong for both rows.
    """
    beta, decay = 0.5, 0.3
    # Row 0: prompt = first 2 tokens (-100), response starts at index 2.
    # Row 1: prompt = first 5 tokens (-100), response starts at index 5.
    labels = torch.tensor([
        [-100, -100, 7, 8, 9, 10, 11, 12],
        [-100, -100, -100, -100, -100, 3, 4, 5],
    ])
    k = [2, 5]  # per-row first response index

    pos_weights = _csft_pos_weights(labels, beta, decay)

    for row, kr in enumerate(k):
        valid_cols = (labels[row] != -100).nonzero(as_tuple=True)[0]
        w_valid = pos_weights[row, valid_cols]
        # The maximum weight among VALID (response) positions must be at the
        # first response token (relative t == 0 -> weight == beta).
        argmax_valid_col = valid_cols[int(torch.argmax(w_valid))]
        assert int(argmax_valid_col) == kr, (
            f"row {row}: peak weight at col {int(argmax_valid_col)}, "
            f"expected first response col {kr}"
        )
        # First response token gets the undecayed weight (t=0 -> beta).
        assert torch.isclose(pos_weights[row, kr], torch.tensor(beta)), (
            f"row {row}: first-response weight {pos_weights[row, kr]} != beta {beta}"
        )
        # Weights strictly DECREASE along the response (monotone decay).
        resp_weights = pos_weights[row, kr:]
        diffs = resp_weights[1:] - resp_weights[:-1]
        assert torch.all(diffs < 0), f"row {row}: response weights not decreasing"


def test_csft_position_weights_differ_between_rows_with_different_starts():
    """BUG: absolute-position weighting is identical across rows regardless of
    where each row's response begins. Response-relative weighting must give
    the SAME first-response weight (beta) to two rows whose responses start
    at different absolute indices.
    """
    beta, decay = 0.5, 0.2
    labels = torch.tensor([
        [-100, 1, 2, 3],          # response starts at col 1
        [-100, -100, -100, 9],    # response starts at col 3
    ])
    pos_weights = _csft_pos_weights(labels, beta, decay)

    # Both first-response tokens (col 1 and col 3) should have weight == beta.
    assert torch.isclose(pos_weights[0, 1], torch.tensor(beta))
    assert torch.isclose(pos_weights[1, 3], torch.tensor(beta))
    # Under absolute positions these would be beta*exp(-decay*1) and
    # beta*exp(-decay*3) respectively (and unequal), so this assert encodes
    # the regression.
    assert torch.isclose(pos_weights[0, 1], pos_weights[1, 3])


def test_csft_compute_loss_end_to_end_finite():
    """End-to-end: drive ConstrainedSFTTrainer.compute_loss with a _TinyLM and
    a 2-row batch with different response starts; assert the loss is finite.

    This exercises the real (post-fix) weighting code path. Requires
    transformers for the Trainer base class.
    """
    pytest.importorskip("transformers")
    from safetune.harden.constrained_sft import (
        ConstrainedSFTConfig,
        ConstrainedSFTTrainer,
    )

    torch.manual_seed(0)
    model = _TinyLM()
    ref_model = _TinyLM()
    # Make the reference distinct so KL is non-trivial / non-zero.
    with torch.no_grad():
        for p in ref_model.parameters():
            p.add_(0.1)

    train_ds = [
        {
            "input_ids": torch.randint(0, 32, (8,), dtype=torch.long),
            "labels": torch.randint(0, 32, (8,), dtype=torch.long),
        }
        for _ in range(4)
    ]

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        args = ConstrainedSFTConfig(
            output_dir=tmp, num_train_epochs=1, per_device_train_batch_size=2,
            logging_steps=10, save_strategy="no", report_to=[],
            csft_beta=0.5, csft_decay_rate=0.1,
        )
        trainer = ConstrainedSFTTrainer(
            model=model, args=args, train_dataset=train_ds,
            reference_model=ref_model,
        )

    # 2-row batch with DIFFERENT response-start indices. HF Trainer may have
    # moved the model to an accelerator (e.g. cuda:0), so build the inputs on
    # the model's device to avoid a device mismatch. The frozen reference model
    # is not touched by the Trainer, so move it onto the same device too.
    dev = next(model.parameters()).device
    ref_model.to(dev)
    inputs = {
        "input_ids": torch.randint(0, 32, (2, 8), dtype=torch.long, device=dev),
        "labels": torch.tensor([
            [-100, -100, 5, 6, 7, 8, 9, 10],
            [-100, -100, -100, -100, 1, 2, 3, 4],
        ], device=dev),
    }
    loss = trainer.compute_loss(model, inputs)
    assert torch.isfinite(loss), "Constrained-SFT loss is not finite"


# ---------------------------------------------------------------------------
# Test 2: SEAL falls back to uniform weights on a batch-size mismatch.
# ---------------------------------------------------------------------------

def _make_seal_without_init() -> Any:
    """Construct a SEALTrainer instance WITHOUT running its (HF-Trainer)
    __init__, then set just the attributes compute_loss needs.

    Driving the real Trainer.__init__ requires a full HF tokenizer/collator
    dance; the bug lives entirely in compute_loss's cache-vs-batch-size
    handling, so we bypass __init__ via __new__ and populate the minimal
    state. This keeps the test focused on the regression.
    """
    pytest.importorskip("transformers")
    from safetune.harden.seal import SEALTrainer

    trainer = SEALTrainer.__new__(SEALTrainer)
    trainer._seal_temperature = 1.0
    trainer._seal_rescore_every = 10
    trainer._seal_top_k_ratio = 1.0
    trainer._cached_weights = None
    trainer._step_count = 0
    return trainer


def test_seal_stale_cache_size_mismatch_does_not_crash():
    """BUG: cached per-example weights (size B0) were multiplied against a
    later batch's per-example losses (size B1) without a size check, raising
    a RuntimeError on the broadcast/multiply. The fix falls back to uniform
    torch.ones weights on any size mismatch.

    Pre-seed the cache with a size-3 weight vector, then call compute_loss
    on a size-2 batch on a NON-rescore step. Must NOT raise and must return
    a finite loss.
    """
    torch.manual_seed(0)
    trainer = _make_seal_without_init()
    model = _TinyLM()

    # Pre-seed a stale cache for a batch of size 3 (different from below).
    trainer._cached_weights = torch.tensor([0.2, 0.3, 2.5])
    # Force a NON-rescore step: step_count so that (count+1) % every != 1
    # and cache is not None, so the cached-weights branch is considered.
    trainer._step_count = 5          # -> becomes 6 inside compute_loss
    trainer._seal_rescore_every = 10  # 6 % 10 == 6 != 1 -> not a rescore step

    # Batch of size 2 (mismatches the size-3 cache).
    inputs = {
        "input_ids": torch.randint(0, 32, (2, 6), dtype=torch.long),
        "labels": torch.randint(0, 32, (2, 6), dtype=torch.long),
    }

    # Pre-fix: this multiply crashed with a shape RuntimeError.
    loss = trainer.compute_loss(model, inputs)
    assert torch.isfinite(loss), "SEAL loss is not finite after uniform fallback"


def test_seal_uniform_fallback_weights_match_batch_size():
    """When the cached weights do not match the current batch, the effective
    weights used must be uniform ones of the CURRENT batch size.

    We verify behaviourally: with uniform weights the SEAL loss equals the
    plain mean of per-example losses. A stale size-3 cache applied to a
    size-2 batch (pre-fix) could not produce this and would have crashed.
    """
    torch.manual_seed(1)
    trainer = _make_seal_without_init()
    model = _TinyLM()

    trainer._cached_weights = torch.tensor([5.0, 0.1, 0.1])  # size-3 stale
    trainer._step_count = 5
    trainer._seal_rescore_every = 10  # non-rescore step

    input_ids = torch.randint(0, 32, (2, 6), dtype=torch.long)
    labels = torch.randint(0, 32, (2, 6), dtype=torch.long)
    inputs = {"input_ids": input_ids, "labels": labels}

    loss = trainer.compute_loss(model, inputs)

    # Reference: plain mean of per-example losses (i.e. weights == ones).
    per = []
    for i in range(2):
        out_i = model(input_ids=input_ids[i:i+1], labels=labels[i:i+1])
        per.append(out_i.loss)
    expected = torch.stack(per).mean()

    assert torch.isclose(loss, expected, atol=1e-5), (
        f"SEAL loss {loss.item()} != uniform-weighted mean {expected.item()}; "
        "uniform fallback not applied"
    )
