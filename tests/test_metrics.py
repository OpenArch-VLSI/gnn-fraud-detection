import torch
from src.utils.metrics import selection_score, budget_threshold, budget_stats


def test_selection_score_is_pr_auc_only():
    assert selection_score({'pr_auc': 0.4, 'roc_auc': 0.9}) == 0.4


def test_budget_flags_about_the_requested_fraction():
    logits = torch.linspace(-3, 3, 1000)
    thr = budget_threshold(logits, 0.01)
    flagged = (logits >= thr).float().mean().item()
    assert abs(flagged - 0.01) < 0.005


def test_budget_precision_and_recall_on_a_known_case():
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    targets = torch.tensor([1, 0, 1, 0, 0])
    stats = budget_stats(logits, targets, threshold=3.5)
    assert stats['flagged_fraction'] == 2 / 5
    assert stats['precision_at_budget'] == 1 / 2
    assert stats['recall_at_budget'] == 1 / 2