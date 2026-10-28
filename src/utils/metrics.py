import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Classification metrics for binary classification. Threshold-free metrics first."""
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    targets_np = targets.cpu().numpy()

    try:
        roc_auc = roc_auc_score(targets_np, probs)
    except ValueError:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(targets_np, probs)
    except ValueError:
        pr_auc = 0.0

    # F1 and recall at the fixed 0.5 threshold are kept for reference only.
    # The operating point that matters is the review budget (see budget_stats).
    f1 = f1_score(targets_np, preds, zero_division=0)
    recall = recall_score(targets_np, preds, zero_division=0)

    return {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'f1': f1, 'recall': recall}


def budget_threshold(logits: torch.Tensor, budget: float) -> float:
    """
    Logit threshold that flags the top `budget` fraction of these scores.
    Compute it on the validation split only.
    """
    if not 0 < budget < 1:
        raise ValueError(f"budget must be in (0, 1), got {budget}")
    scores = logits.detach().cpu().numpy()
    return float(np.quantile(scores, 1.0 - budget))


def budget_stats(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict:
    """Precision and recall when everything at or above `threshold` is flagged."""
    scores = logits.detach().cpu().numpy()
    y = targets.detach().cpu().numpy().astype(int)
    flagged = scores >= threshold
    n_flagged = int(flagged.sum())
    n_pos = int((y == 1).sum())
    tp = int((flagged & (y == 1)).sum())
    return {
        'precision_at_budget': tp / n_flagged if n_flagged else 0.0,
        'recall_at_budget': tp / n_pos if n_pos else 0.0,
        'flagged_fraction': n_flagged / len(y) if len(y) else 0.0,
    }


def selection_score(metrics: dict) -> float:
    """
    Checkpoint selection and early stopping use validation PR-AUC alone.
    PR-AUC is the metric that matches the fraud-detection goal. ROC-AUC is
    reported but does not influence which checkpoint is kept.
    """
    return float(metrics['pr_auc'])


def budget_report(val_logits: torch.Tensor, test_logits: torch.Tensor,
                  test_labels: torch.Tensor, budgets) -> list:
    """
    For each review budget: the threshold chosen on validation, then precision,
    recall and flagged fraction on test using that threshold.
    """
    rows = []
    for b in budgets:
        thr = budget_threshold(val_logits, b)
        stats = budget_stats(test_logits, test_labels, thr)
        rows.append({'budget': b, 'threshold': thr, **stats})
    return rows