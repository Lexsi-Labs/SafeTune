"""Audit U05: FLAT feeds the f-divergence average token *probabilities*, not log-likelihoods.

Wang et al. 2024 (arXiv:2410.11143) Sec. 3.2 / Eq. before 3.3: P(x, y; theta) is the average
over answer tokens of the next-token probability, with no logarithm ("whether to apply the
logarithm to the output probability" is one of the three differences from DPO), and

    L_FLAT = - E[ g*(P(x_f, y_e)) - f*(g*(P(x_f, y_f))) ]

Official UCSC-REAL/FLAT dataloader.py `get_contrastive_loss(prob_sum_unlearn, prob_sum_good)`
receives (negated) mean token probabilities from `ProbLossStable`. flat_fdiv_loss feeds the
negated mean token NLL (log scale) instead.
"""
import math

import torch

from safetune.unlearn import flat_fdiv_loss


def _logits_for(probs, vocab=4):
    """Logits at positions 0..len(probs)-1 so the next token (label 1..) has the given prob."""
    rows = []
    for p in probs:
        rest = (1.0 - p) / (vocab - 1)
        row = [math.log(rest)] * vocab
        row[1] = math.log(p)           # label id 1 is the correct token at every position
        rows.append(row)
    rows.append([0.0] * vocab)          # last position is not scored
    return torch.tensor([rows])


def test_flat_kl_loss_uses_mean_token_probability():
    good_p, forget_p = [0.5, 0.25, 0.25], [0.2, 0.2, 0.4]
    labels = torch.tensor([[-100, 1, 1, 1]])
    loss = flat_fdiv_loss(_logits_for(good_p), labels, _logits_for(forget_p), labels, divergence="kl")
    x_good, x_forget = sum(good_p) / 3, sum(forget_p) / 3
    expected = -x_good + math.exp(x_forget - 1.0)      # g*(P_good) - f*(P_forget), KL
    assert abs(loss.item() - expected) < 1e-5, f"loss {loss.item():.4f}, paper formula {expected:.4f}"


def test_flat_docs_do_not_call_t_a_log_likelihood():
    import pathlib

    from safetune.interventions.unlearn import flat

    doc = flat.__doc__
    assert "per-answer log-likelihood" not in doc
    assert "average next-token probability over the answer tokens" in doc
    guide = pathlib.Path(__file__).parents[2] / "docs/user-guide/unlearn/flat.md"
    if guide.exists():
        assert "log-likelihood" not in guide.read_text()
