"""Regression tests for the unlearn correctness fixes.

These tests target two unlearn correctness areas on the
``fix/runner-and-method-correctness`` branch:

1. ``safetune.interventions.unlearn.flat.flat_fdiv_loss`` — the FLAT objective.  The
   previous implementation was a confidence-weighted gradient *ascent* whose
   sign was inverted (it trained *toward* the forget data) and which was not
   the published FLAT algorithm at all.  FLAT (Wang et al., ICLR 2025,
   arXiv:2410.11143) is an f-divergence *loss adjustment*: it pairs each forget
   answer with a template/good (refusal) answer and maximises an f-divergence
   between them, pushing the good answer's likelihood UP and the forget
   answer's likelihood DOWN.  These tests assert exactly that directional
   behaviour plus finiteness across the validated divergence variants.

2. ``safetune.interventions.unlearn.simdpo.make_simdpo_pairs`` — the pad-masking +
   prompt-boundary fix.  Post-fix every pad position in the chosen labels is
   ``-100``, and the prompt span is masked using a consistent re-tokenization
   of the prompt (not a count borrowed from the harmful batch's tokenization).
"""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# 1. FLAT confidence weighting.
# ---------------------------------------------------------------------------

# The directionally-stable divergences validated across operating regimes.
_STABLE_DIVERGENCES = ["kl", "jensen_shannon", "total_variation", "jeffrey"]


def _toy_answer_logits(vocab=12, seq=6, n_prompt=2, seed=0):
    """Random (logits, labels) for one example whose first ``n_prompt`` tokens
    are masked (-100) and the rest are answer targets."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, seq, vocab, generator=g).requires_grad_(True)
    labels = torch.randint(0, vocab, (1, seq), generator=g)
    labels[0, :n_prompt] = -100
    return logits, labels


@pytest.mark.parametrize("div", _STABLE_DIVERGENCES)
def test_flat_fdiv_loss_is_finite(div):
    """The FLAT f-divergence objective must be a finite scalar for the
    validated divergence variants."""
    from safetune.interventions.unlearn.flat import flat_fdiv_loss

    gl, glab = _toy_answer_logits(seed=1)
    fl, flab = _toy_answer_logits(seed=2)
    loss = flat_fdiv_loss(gl, glab, fl, flab, divergence=div)
    assert loss.dim() == 0
    assert torch.isfinite(loss), f"{div} loss not finite: {loss}"


@pytest.mark.parametrize("div", _STABLE_DIVERGENCES)
def test_flat_fdiv_pushes_good_up_and_forget_down(div):
    """Core FLAT property: minimising the loss must INCREASE the good/template
    answer's likelihood and DECREASE the forget answer's likelihood.

    This guards the previous implementation's catastrophic sign inversion,
    which trained the model *toward* the forget data.  We check the gradient on
    the correct-class logit of an answer token: gradient descent moves a logit
    by ``-grad``, so the good answer's correct logit must move UP (grad < 0)
    and the forget answer's correct logit must move DOWN (grad > 0).
    """
    from safetune.interventions.unlearn.flat import flat_fdiv_loss

    gl, glab = _toy_answer_logits(seed=3)
    fl, flab = _toy_answer_logits(seed=4)
    loss = flat_fdiv_loss(gl, glab, fl, flab, divergence=div)
    loss.backward()

    # First answer position is index n_prompt-1 (predicts label at n_prompt).
    pos = 1
    good_idx = glab[0, pos + 1].item()
    forget_idx = flab[0, pos + 1].item()
    good_grad = gl.grad[0, pos, good_idx].item()
    forget_grad = fl.grad[0, pos, forget_idx].item()

    assert good_grad < 0, (
        f"[{div}] good-answer correct logit should move UP (grad<0), "
        f"got {good_grad}"
    )
    assert forget_grad > 0, (
        f"[{div}] forget-answer correct logit should move DOWN (grad>0), "
        f"got {forget_grad}"
    )


def test_flat_fdiv_rejects_unknown_divergence():
    from safetune.interventions.unlearn.flat import flat_fdiv_loss

    gl, glab = _toy_answer_logits(seed=5)
    fl, flab = _toy_answer_logits(seed=6)
    with pytest.raises(ValueError):
        flat_fdiv_loss(gl, glab, fl, flab, divergence="not-a-divergence")


def test_flat_unlearn_requires_good_batches():
    """flat_unlearn must refuse to run without template/good answers (the
    forget-only-ascent footgun the old API allowed)."""
    from safetune.interventions.unlearn.flat import flat_unlearn, FLATConfig

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, input_ids=None, **kw):
            h = torch.nn.functional.one_hot(input_ids.long(), 4).float()
            return type("O", (), {"logits": self.lin(h)})

    forget = [{"input_ids": torch.tensor([[0, 1, 2, 3]]),
               "attention_mask": torch.ones(1, 4, dtype=torch.long),
               "labels": torch.tensor([[-100, 1, 2, 3]])}]
    with pytest.raises(ValueError):
        flat_unlearn(_Tiny(), forget, good_batches=None,
                     config=FLATConfig(variant="flat"))


# ---------------------------------------------------------------------------
# 2. SimDPO pair construction: pad masking + prompt boundary.
# ---------------------------------------------------------------------------

class _StubTokenizer:
    """Deterministic word-level stub tokenizer with a pad token.

    Whitespace-splits text into words and maps each to a stable integer id via
    a growing vocab.  Supports the subset of the HF tokenizer API that
    ``make_simdpo_pairs`` exercises:

      * ``tokenizer(text, return_tensors="pt", max_length=..., truncation=...,
        padding="max_length" | (absent), add_special_tokens=...)`` -> dict with
        ``input_ids`` and ``attention_mask`` ``(1, L)`` tensors.
      * ``tokenizer.decode(ids, skip_special_tokens=True)`` -> str.

    A single leading special (BOS) token id is prepended when
    ``add_special_tokens`` is True (the default), so prompt re-encoding length
    is realistic.  The pad id is distinct from every word id.
    """

    pad_token = "<pad>"
    pad_token_id = 0
    bos_token_id = 1

    def __init__(self) -> None:
        # Reserve 0 for pad and 1 for BOS; real words start at 2.
        self._vocab = {self.pad_token: 0, "<bos>": 1}

    def _id(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = len(self._vocab)
        return self._vocab[word]

    def __call__(
        self,
        text,
        return_tensors=None,
        max_length=256,
        truncation=False,
        padding=None,
        add_special_tokens=True,
    ):
        words = text.split()
        ids = [self.bos_token_id] if add_special_tokens else []
        ids += [self._id(w) for w in words]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if padding == "max_length" and max_length is not None:
            pad_n = max_length - len(ids)
            attn = [1] * len(ids) + [0] * pad_n
            ids = ids + [self.pad_token_id] * pad_n
        else:
            attn = [1] * len(ids)
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.tensor([attn], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        inv = {v: k for k, v in self._vocab.items()}
        out = []
        for i in ids.tolist() if torch.is_tensor(ids) else list(ids):
            word = inv.get(int(i), "")
            if skip_special_tokens and word in (self.pad_token, "<bos>"):
                continue
            out.append(word)
        return " ".join(out)


def _make_harmful_batch(tok: _StubTokenizer, prompt: str, response: str):
    """Build a single-example harmful batch with the prompt span masked.

    Encodes ``prompt + " " + response`` and sets ``labels = -100`` over the
    prompt token span (HF convention), leaving the response tokens as targets.
    """
    full = prompt + " " + response
    enc = tok(full, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"]
    # Number of prompt tokens when encoded WITH special tokens.
    prompt_enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    prompt_len = prompt_enc["input_ids"].shape[1]
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


def test_make_simdpo_pairs_pads_masked_in_chosen_labels():
    """Every padded position in the chosen sequence must be ``-100``.

    Bug guarded (a): with ``padding="max_length"`` the trailing PAD tokens were
    left as cross-entropy targets, training the model to emit pad.  Post-fix
    all positions where ``attention_mask == 0`` (equivalently ``input_ids ==
    pad_token_id`` in the padded tail) are masked to ``-100``.
    """
    from safetune.interventions.unlearn.simdpo import make_simdpo_pairs

    tok = _StubTokenizer()
    harmful = [_make_harmful_batch(tok, "tell me how", "harmful answer here")]
    refusal = "I cannot help with that"

    max_len = 32
    pairs = make_simdpo_pairs(harmful, refusal, tok, max_len=max_len)
    assert len(pairs) == 1

    chosen = pairs[0]["chosen"]
    input_ids = chosen["input_ids"][0]
    labels = chosen["labels"][0]

    # With padding="max_length" the chosen sequence is padded to max_len.
    assert input_ids.shape[0] == max_len

    pad_positions = input_ids == tok.pad_token_id
    assert pad_positions.any(), "expected some pad positions to exercise the fix"
    # Bug (a): every pad position must be masked out of the loss.
    assert (labels[pad_positions] == -100).all(), (
        "pad positions in chosen labels must be -100, not CE targets"
    )


def test_make_simdpo_pairs_prompt_boundary_uses_consistent_tokenization():
    """The number of leading masked (-100) chosen labels must equal the prompt
    token count under the SAME tokenization used for ``chosen_text``.

    Bug guarded (b): the prompt boundary was taken as ``len(prompt_ids)`` from
    the harmful batch's tokenization, which can differ from the chosen-text
    tokenization (special tokens / settings), masking the wrong span.  Post-fix
    the prompt is re-encoded consistently and its token count is the boundary.
    """
    from safetune.interventions.unlearn.simdpo import make_simdpo_pairs

    tok = _StubTokenizer()
    prompt = "tell me how"
    harmful = [_make_harmful_batch(tok, prompt, "harmful answer here")]
    refusal = "I cannot help with that"

    pairs = make_simdpo_pairs(harmful, refusal, tok, max_len=32)
    labels = pairs[0]["chosen"]["labels"][0]

    # Leading run of -100 == masked prompt span.
    n_leading_masked = 0
    for v in labels.tolist():
        if v == -100:
            n_leading_masked += 1
        else:
            break

    # Recover the prompt text the way make_simdpo_pairs does (decode the
    # harmful batch's prompt span) and re-encode with consistent settings.
    h_input = harmful[0]["input_ids"][0]
    h_labels = harmful[0]["labels"][0]
    prompt_text = tok.decode(h_input[h_labels == -100], skip_special_tokens=True)
    prompt_enc = tok(
        prompt_text, return_tensors="pt", max_length=32,
        truncation=True, add_special_tokens=True,
    )
    expected_prompt_len = prompt_enc["input_ids"].shape[1]

    # Bug (b): the masked prefix length must match the consistent prompt
    # tokenization exactly (and stay clear of the padded tail).
    assert n_leading_masked == expected_prompt_len, (
        f"masked prompt prefix {n_leading_masked} should equal consistent "
        f"prompt length {expected_prompt_len}"
    )
    # The masked prefix must be a real prompt boundary, not the whole sequence
    # (which would happen if pad-masking collapsed everything).
    assert n_leading_masked < labels.shape[0]
