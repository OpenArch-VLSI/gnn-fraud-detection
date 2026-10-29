"""
Surrogates for the two metrics that checkpoint selection is judged on.

train.py trains with weighted BCE, which fits calibrated probabilities but
does not optimise ranking directly. The two terms here are computed on the
same seed nodes as the BCE loss and added to it:

- pairwise_auc_loss: mean logistic loss over every (fraud, legitimate) pair
  in the batch. Minimising it pushes the batch ROC-AUC towards 1.
- smooth_ap_loss: 1 - average precision, with the ranks made differentiable
  using sigmoids. Minimising it pushes the batch PR-AUC towards 1.

Both take logits. When a batch cannot define the metric (no positives, or
no negatives for the pairwise term) they return a zero that still has a
gradient path, so such a batch adds nothing to the loss.
"""

import torch
import torch.nn.functional as F


def pairwise_auc_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Mean over all (positive, negative) pairs of log(1 + exp(-(s_pos - s_neg))).

    Each term is near 0 when the fraud case is scored above the legitimate
    one and grows roughly linearly with the size of the inversion. Averaging
    over pairs gives a convex surrogate for (1 - ROC-AUC) on the batch.
    """
    is_pos = labels.to(logits.dtype) > 0.5
    pos = logits[is_pos]
    neg = logits[~is_pos]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.sum() * 0.0
    margins = pos[:, None] - neg[None, :]
    return F.softplus(-margins).mean()


def smooth_ap_loss(logits: torch.Tensor, labels: torch.Tensor,
                   temperature: float = 1.0) -> torch.Tensor:
    """
    1 - average precision, with the ranking made differentiable.

    For each positive i, the number of samples scored at or above it is
    estimated as 1 + sum_j sigmoid((s_j - s_i) / T) over the other samples,
    and the number of positives among them the same way. Precision at i is
    the ratio of the two, and AP is the mean precision over the positives.
    With no tied scores this is exactly average_precision_score's definition.
    As T -> 0 the sigmoids become step functions and the value becomes exact.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    labels = labels.to(logits.dtype)
    pos_idx = torch.nonzero(labels > 0.5, as_tuple=True)[0]
    if pos_idx.numel() == 0:
        return logits.sum() * 0.0

    # above[i, j] is close to 1 when sample j is scored above positive i.
    above = torch.sigmoid((logits[None, :] - logits[pos_idx, None]) / temperature)
    # Remove each positive's comparison with itself, which is sigmoid(0) = 0.5.
    self_mask = torch.zeros_like(above, dtype=torch.bool)
    self_mask[torch.arange(pos_idx.numel(), device=logits.device), pos_idx] = True
    above = above.masked_fill(self_mask, 0.0)

    rank = 1.0 + above.sum(dim=1)
    positives_above = 1.0 + (above * labels[None, :]).sum(dim=1)
    return 1.0 - (positives_above / rank).mean()
