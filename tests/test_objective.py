import os
import sys

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.losses import pairwise_auc_loss, smooth_ap_loss
from src.utils.metrics import selection_score


def _brute_force_pairwise(logits, labels):
    terms = [np.log1p(np.exp(-(logits[i] - logits[j])))
             for i in range(len(labels)) for j in range(len(labels))
             if labels[i] == 1 and labels[j] == 0]
    return float(np.mean(terms))


def test_pairwise_loss_matches_brute_force_over_pairs():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=30)
    labels = (rng.random(30) < 0.3).astype(int)
    labels[0], labels[1] = 1, 0
    got = pairwise_auc_loss(torch.tensor(logits, dtype=torch.float32),
                            torch.tensor(labels))
    assert got.item() == pytest.approx(_brute_force_pairwise(logits, labels), rel=1e-5)


def test_pairwise_loss_small_when_fraud_is_scored_above_legitimate():
    logits = torch.tensor([3.0, 4.0, -3.0, -2.0, -4.0])
    labels = torch.tensor([1, 1, 0, 0, 0])
    assert pairwise_auc_loss(logits, labels).item() < 0.01


def test_pairwise_loss_large_when_ranking_is_reversed():
    logits = torch.tensor([-3.0, -4.0, 3.0, 2.0])
    labels = torch.tensor([1, 1, 0, 0])
    assert pairwise_auc_loss(logits, labels).item() > 5.0


def test_pairwise_loss_is_zero_when_batch_has_one_class():
    logits = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
    for labels in (torch.zeros(3, dtype=torch.long), torch.ones(3, dtype=torch.long)):
        loss = pairwise_auc_loss(logits, labels)
        assert loss.item() == 0.0
        loss.backward()
        assert torch.all(logits.grad == 0)
        logits.grad = None


def test_smooth_ap_is_small_for_perfect_ranking():
    logits = torch.tensor([4.0, 5.0, -4.0, -5.0, -6.0, -4.5])
    labels = torch.tensor([1, 1, 0, 0, 0, 0])
    assert smooth_ap_loss(logits, labels).item() < 0.01


def test_smooth_ap_is_large_for_reversed_ranking():
    logits = torch.tensor([-4.0, -5.0, 4.0, 5.0, 6.0, 4.5])
    labels = torch.tensor([1, 1, 0, 0, 0, 0])
    assert smooth_ap_loss(logits, labels).item() > 0.5


def test_smooth_ap_converges_to_exact_average_precision():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=60)
    labels = (rng.random(60) < 0.25).astype(int)
    labels[0], labels[1] = 1, 0
    exact_ap = average_precision_score(labels, logits)
    got = smooth_ap_loss(torch.tensor(logits, dtype=torch.float32),
                         torch.tensor(labels), temperature=1e-4)
    assert got.item() == pytest.approx(1.0 - exact_ap, abs=1e-3)


def test_smooth_ap_is_zero_without_positives():
    logits = torch.tensor([0.1, -0.2, 0.3], requires_grad=True)
    loss = smooth_ap_loss(logits, torch.zeros(3, dtype=torch.long))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.all(logits.grad == 0)


def test_smooth_ap_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        smooth_ap_loss(torch.zeros(2), torch.tensor([1, 0]), temperature=0.0)


def test_both_surrogates_give_finite_nonzero_gradients():
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(40, generator=gen, requires_grad=True)
    labels = (torch.rand(40, generator=gen) < 0.2).long()
    labels[0], labels[1] = 1, 0
    total = pairwise_auc_loss(logits, labels) + smooth_ap_loss(logits, labels)
    total.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0


def test_resume_refuses_checkpoint_trained_under_a_different_objective():
    import src.train as T

    current = {'model': 'sage', 'hidden_channels': 64,
               'rank_loss_weight': 1.0, 'ap_loss_weight': 1.0, 'ap_temperature': 1.0}
    T.validate_resume_config(dict(current), current)  # identical configs pass

    # A config written before the objective change has no objective fields.
    pre_change = {'model': 'sage', 'hidden_channels': 64}
    with pytest.raises(ValueError, match='rank_loss_weight'):
        T.validate_resume_config(pre_change, current)

    switched = dict(current, ap_loss_weight=0.0)
    with pytest.raises(ValueError, match='ap_loss_weight'):
        T.validate_resume_config(current, switched)
