"""
Non-graph reference baseline: LightGBM on the node feature matrix only.

Uses the same time-based masks and features as the graph models. Edges are
ignored, so the causal and full graph files give the same result. Reports
test metrics in the same format as evaluate.py.

Usage:
    python src/train_tabular.py --config configs/lgbm_tabular.yaml
"""
import argparse
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.metrics import compute_metrics, budget_report
from src.utils.seed import set_seed

DEFAULT_BUDGETS = [0.01, 0.05, 0.10]
DEFAULT_PARAMS = {
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
}


def train_lightgbm(x, y, train_mask, val_mask, params, num_boost_round,
                   early_stopping_rounds, seed):
    """
    Trains on the training split and stops early on validation PR-AUC.
    Class imbalance is handled with scale_pos_weight, computed on the training split.
    Returns the booster and the best iteration count.
    """
    x_train, y_train = x[train_mask], y[train_mask]
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos

    full_params = dict(DEFAULT_PARAMS)
    full_params.update(params)
    full_params.update({
        'objective': 'binary',
        'metric': 'average_precision',
        'seed': seed,
        'verbosity': -1,
        'scale_pos_weight': float(n_neg / max(n_pos, 1)),
    })

    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x[val_mask], label=y[val_mask], reference=dtrain)
    booster = lgb.train(
        full_params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        valid_names=['val'],
        callbacks=[lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False)],
    )
    best = booster.best_iteration or booster.current_iteration()
    return booster, best


def run_experiment(x, y, masks, cfg, exp_dir):
    """Train, then write config, model, metrics and test results to exp_dir."""
    set_seed(cfg['seed'])
    booster, best = train_lightgbm(
        x, y, masks['train'], masks['val'],
        params=cfg.get('params', {}),
        num_boost_round=cfg['num_boost_round'],
        early_stopping_rounds=cfg['early_stopping_rounds'],
        seed=cfg['seed'],
    )

    val_logits = torch.tensor(booster.predict(x[masks['val']], raw_score=True, num_iteration=best))
    test_logits = torch.tensor(booster.predict(x[masks['test']], raw_score=True, num_iteration=best))
    test_labels = torch.tensor(y[masks['test']])
    budgets = cfg.get('review_budgets', DEFAULT_BUDGETS)

    val_metrics = compute_metrics(val_logits, torch.tensor(y[masks['val']]))
    test_results = compute_metrics(test_logits, test_labels)
    test_results['budgets'] = budget_report(val_logits, test_logits, test_labels, budgets)

    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=4)
    booster.save_model(os.path.join(exp_dir, 'model.txt'), num_iteration=best)
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump({'best_iteration': best, 'val': val_metrics}, f, indent=4)
    with open(os.path.join(exp_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)

    print(f"best_iteration={best}")
    print(f"validation PR-AUC {val_metrics['pr_auc']:.4f}, ROC-AUC {val_metrics['roc_auc']:.4f}")
    print(f"test       PR-AUC {test_results['pr_auc']:.4f}, ROC-AUC {test_results['roc_auc']:.4f}")
    for row in test_results['budgets']:
        print(f"  budget {row['budget']:.0%}: precision {row['precision_at_budget']:.4f}, "
              f"recall {row['recall_at_budget']:.4f}")
    return test_results


def main():
    parser = argparse.ArgumentParser(description="LightGBM tabular baseline (no graph)")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault('seed', 42)
    cfg.setdefault('num_boost_round', 2000)
    cfg.setdefault('early_stopping_rounds', 100)
    cfg.setdefault('run_name', 'lgbm_tabular')
    cfg.setdefault('graph', 'causal')

    data_path = os.path.join('data', 'processed', f"graph_{cfg['graph']}.pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Graph not found at {data_path}. Run src/data/build_graph.py first.")
    data = torch.load(data_path, weights_only=False)
    x = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    masks = {
        'train': data.train_mask.cpu().numpy(),
        'val': data.val_mask.cpu().numpy(),
        'test': data.test_mask.cpu().numpy(),
    }
    print(f"Loaded {x.shape[0]} transactions x {x.shape[1]} features (edges ignored).")

    run_experiment(x, y, masks, cfg, os.path.join('experiments', cfg['run_name']))


if __name__ == "__main__":
    main()