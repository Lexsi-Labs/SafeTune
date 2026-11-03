"""Audit U05 (review of #110): every FLAT divergence has finite loss and gradient at p in [0, 1].

The mean token probability reaches 0 and 1. neyman's f*(x) = 2 - 2*sqrt(1 - x) has an infinite
gradient at x = 1 (it was finite when T was a log-likelihood, x <= 0). The loss must still equal
the official get_contrastive_loss.
"""
import math

import pytest
import torch

from safetune.interventions.unlearn import flat as F

EDGES = (0.0, 1e-8, 0.5, 1 - 1e-8, 1.0)


@pytest.mark.parametrize("div", F._DIVERGENCES)
@pytest.mark.parametrize("p", EDGES)
def test_loss_and_gradients_are_finite_at_edge_probabilities(div, p):
    g, fstar = F._g_fstar(div)
    good = torch.tensor([p], requires_grad=True)
    forget = torch.tensor([p], requires_grad=True)
    loss = g(good) - fstar(forget)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(good.grad).all() and torch.isfinite(forget.grad).all()


# Verbatim from UCSC-REAL/FLAT dataloader.py:806-860 @bf2ef07 (activation/conjugate bodies).
def _official(prob_sum_unlearn, prob_sum_good, div):
    if div == 'KL':
        def activation(x): return -torch.mean(x)
        def conjugate(x): return -torch.mean(torch.exp(x - 1.))
    elif div == 'Reverse-KL':
        def activation(x): return -torch.mean(-torch.exp(x))
        def conjugate(x): return -torch.mean(-1. - x)
    elif div == 'Jeffrey':
        def activation(x): return -torch.mean(x)
        def conjugate(x): return -torch.mean(x + torch.mul(x, x) / 4. + torch.mul(torch.mul(x, x), x) / 16.)
    elif div == 'Squared-Hellinger':
        def activation(x): return -torch.mean(1. - torch.exp(x))
        def conjugate(x): return -torch.mean((1. - torch.exp(x)) / (torch.exp(x)))
    elif div == 'Pearson':
        def activation(x): return -torch.mean(x)
        def conjugate(x): return -torch.mean(torch.mul(x, x) / 4. + x)
    elif div == 'Neyman':
        def activation(x): return -torch.mean(1. - torch.exp(x))
        def conjugate(x): return -torch.mean(2. - 2. * torch.sqrt(1. - x))
    elif div == 'Jenson-Shannon':
        def activation(x): return -torch.mean(- torch.log(1. + torch.exp(-x))) - torch.log(torch.tensor(2.))
        def conjugate(x): return -torch.mean(x + torch.log(1. + torch.exp(-x))) + torch.log(torch.tensor(2.))
    elif div == 'Total-Variation':
        def activation(x): return -torch.mean(torch.tanh(x) / 2.)
        def conjugate(x): return -torch.mean(torch.tanh(x) / 2.)
    return activation(-prob_sum_good) - conjugate(-prob_sum_unlearn)


_NAMES = {"kl": "KL", "reverse_kl": "Reverse-KL", "jeffrey": "Jeffrey",
          "squared_hellinger": "Squared-Hellinger", "pearson": "Pearson", "neyman": "Neyman",
          "jensen_shannon": "Jenson-Shannon", "total_variation": "Total-Variation"}


@pytest.mark.parametrize("div", F._DIVERGENCES)
def test_loss_equals_the_official_get_contrastive_loss(div):
    torch.manual_seed(0)
    labels = torch.tensor([[-100, 1, 2, 3, 4]]).repeat(4, 1)
    for _ in range(5):
        good, forget = torch.randn(4, 5, 8), torch.randn(4, 5, 8)
        ours = F.flat_fdiv_loss(good, labels, forget, labels, divergence=div)
        pg = F._seq_mean_prob(good, labels)
        pf = F._seq_mean_prob(forget, labels)
        # ProbLossStable returns -p; get_contrastive_loss negates it again.
        ref = _official(-pf, -pg, _NAMES[div])
        assert math.isclose(ours.item(), ref.item(), abs_tol=5e-6), (div, ours.item(), ref.item())
