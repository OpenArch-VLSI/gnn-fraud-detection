"""
Paired bootstrap comparison of two models on the same test rows.

Usage:
    python scripts/paired_bootstrap.py \
        --gnn experiments/camo_final_seed42/test_scores.npz \
        --base experiments/lgbm_causal_seed42/test_scores.npz

Resamples test rows with replacement, using the same indices for both models, and
reports the PR-AUC difference (GNN minus baseline) with a 95% interval.
"""
import argparse

import numpy as np
from sklearn.metrics import average_precision_score


def paired_bootstrap_pr_diff(y, s_gnn, s_base, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        diffs.append(average_precision_score(y[idx], s_gnn[idx])
                     - average_precision_score(y[idx], s_base[idx]))
    d = np.array(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {'mean_diff': float(d.mean()), 'ci95': (float(lo), float(hi)),
            'p_gnn_better': float((d > 0).mean()), 'n_boot_used': len(d)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn', required=True, help="test_scores.npz of the GNN run")
    parser.add_argument('--base', required=True, help="test_scores.npz of the baseline run")
    parser.add_argument('--n_boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    g = np.load(args.gnn)
    b = np.load(args.base)
    if g['labels'].shape != b['labels'].shape or not np.array_equal(g['labels'], b['labels']):
        raise ValueError("The two runs do not share the same test rows. Check the graph and split.")

    result = paired_bootstrap_pr_diff(g['labels'], g['scores'], b['scores'],
                                      n_boot=args.n_boot, seed=args.seed)
    lo, hi = result['ci95']
    print(f"PR-AUC difference (GNN - baseline): {result['mean_diff']:+.4f}")
    print(f"95% interval: [{lo:+.4f}, {hi:+.4f}]")
    print(f"GNN better in {result['p_gnn_better']:.1%} of resamples")
    verdict = "GNN wins" if lo > 0 else "no win: interval includes zero or favours the baseline"
    print(f"Decision rule: {verdict}")


if __name__ == "__main__":
    main()