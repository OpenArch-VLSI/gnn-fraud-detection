import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_tabular import run_experiment


def _synthetic(n=3000, n_feat=12, pos_rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_feat))
    # Label depends on the first two features, plus noise, so it is learnable.
    score = x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.5, size=n)
    threshold = np.quantile(score, 1 - pos_rate)
    y = (score > threshold).astype(np.int64)
    idx = np.arange(n)
    masks = {
        'train': idx < int(0.7 * n),
        'val': (idx >= int(0.7 * n)) & (idx < int(0.85 * n)),
        'test': idx >= int(0.85 * n),
    }
    return x, y, masks


def test_lightgbm_baseline_learns_signal_and_writes_outputs(tmp_path):
    x, y, masks = _synthetic()
    cfg = {
        'seed': 42,
        'num_boost_round': 200,
        'early_stopping_rounds': 20,
        'params': {'learning_rate': 0.1, 'num_leaves': 15, 'min_data_in_leaf': 20},
    }
    exp_dir = tmp_path / 'lgbm_tabular'
    test_metrics = run_experiment(x, y, masks, cfg, str(exp_dir))

    # Base rate is ~0.10 on the test split. A model that actually reads the
    # features should be far above that.
    assert test_metrics['pr_auc'] > 0.5, test_metrics
    assert test_metrics['roc_auc'] > 0.8, test_metrics

    for name in ('config.json', 'metrics.json', 'test_results.json', 'model.txt'):
        assert (exp_dir / name).exists(), name
    saved = json.loads((exp_dir / 'test_results.json').read_text())
    assert set(saved) >= {'pr_auc', 'roc_auc', 'f1', 'recall'}


def test_lightgbm_baseline_is_deterministic_for_fixed_seed(tmp_path):
    x, y, masks = _synthetic(seed=1)
    cfg = {'seed': 7, 'num_boost_round': 50, 'early_stopping_rounds': 10,
           'params': {'learning_rate': 0.1, 'num_leaves': 15, 'min_data_in_leaf': 20}}
    a = run_experiment(x, y, masks, cfg, str(tmp_path / 'a'))
    b = run_experiment(x, y, masks, cfg, str(tmp_path / 'b'))
    assert a == b
