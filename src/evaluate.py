import argparse
import json
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import NeighborLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.models import FraudGAT, FraudCamouflageGNN, GraphSAGEModel
from src.utils.metrics import compute_metrics, budget_report

FANOUT = [25, 10]          # same as train.py
SAMPLE_SEED = 0            # same neighbours on every evaluation
DEFAULT_BUDGETS = [0.01, 0.05, 0.10]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split")
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--budgets', type=float, nargs='+', default=DEFAULT_BUDGETS,
                        help="Review budgets as fractions, e.g. 0.01 0.05 0.10")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite an existing test_results.json. Use only if no test "
                             "numbers from this run have informed a decision.")
    return parser.parse_args()


def build_model(config, in_channels):
    hidden = config.get('hidden_channels', 64)
    heads = config.get('heads', 4)
    dropout = config.get('dropout', 0.3)
    if config['model'] == 'sage':
        return GraphSAGEModel(in_channels, hidden)
    if config['model'] == 'gat':
        return FraudGAT(in_channels, hidden, heads=heads, dropout=dropout)
    if config['model'] == 'camouflage':
        return FraudCamouflageGNN(in_channels, hidden, heads=heads, dropout=dropout,
                                  detach_trust=config.get('detach_trust', True))
    raise ValueError(f"Unknown model type: {config['model']}")


@torch.no_grad()
def predict(model, data, mask, device):
    """Predict target nodes with repeatable neighbor sampling and progress logs."""
    import time

    model.eval()
    total_nodes = int(mask.sum().item())
    start_time = time.time()

    print(f"Starting prediction for {total_nodes:,} target nodes.", flush=True)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SAMPLE_SEED)
        loader = NeighborLoader(
            data,
            num_neighbors=FANOUT,
            batch_size=2048,
            input_nodes=mask,
            shuffle=False,
            num_workers=0
        )

        logits, labels = [], []
        processed = 0

        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index).squeeze(-1)

            # Keep predictions and labels for target nodes only, not sampled neighbors.
            logits.append(out[:batch.batch_size].cpu())
            labels.append(batch.y[:batch.batch_size].cpu())
            processed += batch.batch_size

            if batch_idx == 1 or batch_idx % 5 == 0 or processed >= total_nodes:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0.0
                percentage = 100.0 * processed / max(total_nodes, 1)

                print(
                    f"  Batch {batch_idx}: "
                    f"{processed:,}/{total_nodes:,} target nodes "
                    f"({percentage:.1f}%), "
                    f"elapsed={elapsed:.1f}s, "
                    f"rate={rate:.1f} target nodes/s",
                    flush=True
                )

    return torch.cat(logits), torch.cat(labels)

def best_training_epoch(exp_dir):
    """The epoch training saved as best, read from metrics.json. None if unavailable."""
    path = os.path.join(exp_dir, 'metrics.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        log = json.load(f)
    if not log:
        return None
    metric_key = ('contaminated_test_pr_auc'
                  if log[0].get('selection_metric') == 'contaminated_test_pr_auc'
                  or log[0].get('contaminated_test_pr_auc') is not None else 'val_score')
    best = max(log, key=lambda row: row[metric_key])
    return {'epoch': best['epoch'],
            'logged_val_pr_auc': best['pr_auc'],
            'selection_metric': metric_key,
            'contaminated_test_pr_auc': best.get('contaminated_test_pr_auc'),
            'epochs_logged': len(log)}


def validate_contamination_metadata(config, leakage_indices, test_global_ids):
    """Reject a purported full-mix run unless its provenance is complete and exact."""
    required = {
        'contaminated_same_label_eval': True,
        'evaluation_status': 'CONTAMINATED_SAME_LABEL_EVALUATION',
        'leakage_test_fraction': 1.0,
        'training_splits': ['train', 'test'],
        'evaluation_split': 'test',
        'valid_generalization_estimate': False,
    }
    mismatches = [key for key, value in required.items() if config.get(key) != value]
    if config.get('graph') != 'causal':
        mismatches.append('graph')
    if mismatches:
        raise ValueError("Inconsistent CONTAMINATED_SAME_LABEL_EVALUATION metadata: "
                         + ", ".join(mismatches))
    if set(leakage_indices.tolist()) != set(test_global_ids.tolist()) or len(leakage_indices) != len(test_global_ids):
        raise ValueError("CONTAMINATED_SAME_LABEL_EVALUATION requires every original test node "
                         "in leakage_test_indices.pt exactly once.")


def is_contaminated_same_label_config(config):
    """Classify full-mix provenance without allowing ambiguous 100% leakage runs."""
    flag = config.get('contaminated_same_label_eval', False)
    status = config.get('evaluation_status')
    fraction = float(config.get('leakage_test_fraction', 0.0))
    claims_contamination = bool(flag) or status is not None
    if fraction == 1.0 and not claims_contamination:
        raise ValueError(
            "leakage_test_fraction == 1.0 requires explicit "
            "contaminated_same_label_eval metadata; refusing to classify it as clean.")
    if claims_contamination:
        if not flag or status != 'CONTAMINATED_SAME_LABEL_EVALUATION':
            raise ValueError(
                "Contaminated experiment metadata is incomplete: set "
                "contaminated_same_label_eval=true and evaluation_status="
                "CONTAMINATED_SAME_LABEL_EVALUATION.")
        return True
    return False


def load_leakage_indices(exp_dir, configured_fraction, contaminated):
    """Load the recorded exposed test IDs, requiring the artifact for contaminated runs."""
    path = os.path.join(exp_dir, 'leakage_test_indices.pt')
    if os.path.exists(path):
        return torch.load(path, map_location='cpu', weights_only=False).long().view(-1)
    if contaminated:
        raise FileNotFoundError(
            f"CONTAMINATED_SAME_LABEL_EVALUATION requires {path}; refusing to evaluate "
            "without the full exposed-test-ID artifact.")
    if configured_fraction > 0:
        raise FileNotFoundError(
            f"{path} is missing even though config.json specifies a non-zero "
            "leakage_test_fraction. Refusing to report ambiguous results.")
    return torch.empty(0, dtype=torch.long)


def main():
    args = parse_args()
    out_path = os.path.join(args.exp_dir, 'test_results.json')
    scores_path = os.path.join(args.exp_dir, 'test_scores.npz')
    config_path = os.path.join(args.exp_dir, 'config.json')
    model_path = os.path.join(args.exp_dir, 'best_model.pt')

    # Protect both outputs from accidental overwrite. Evaluation should be
    # reproducible and each run's original test result should remain traceable.
    existing_outputs = [p for p in (out_path, scores_path) if os.path.exists(p)]
    if existing_outputs and not args.force:
        raise FileExistsError(
            "Evaluation output already exists: " + ", ".join(existing_outputs) +
            ". Use a new experiment directory, or pass --force only if you "
            "deliberately intend to replace these results."
        )
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Experiment directory must contain config.json and best_model.pt")

    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    graph_name = config.get('graph', 'causal')
    data = torch.load(os.path.join('data', 'processed', f'graph_{graph_name}.pt'), weights_only=False)
    print(f"Graph: {graph_name}")
    contaminated = is_contaminated_same_label_config(config)

    configured_fraction = float(config.get('leakage_test_fraction', 0.0))
    leakage_indices = load_leakage_indices(
        args.exp_dir, configured_fraction, contaminated)

    test_global_ids = torch.where(data.test_mask)[0].cpu()
    test_id_set = set(test_global_ids.tolist())
    leakage_list = leakage_indices.tolist()
    if len(leakage_list) != len(set(leakage_list)):
        raise ValueError("leakage_test_indices.pt contains duplicate node IDs.")
    if any(node_id not in test_id_set for node_id in leakage_list):
        raise ValueError("Leakage indices include nodes outside the test split.")
    expected_leak_count = int(round(configured_fraction * len(test_global_ids)))
    if len(leakage_list) != expected_leak_count:
        raise ValueError(
            f"Leakage index count mismatch: config expects {expected_leak_count} "
            f"test nodes, but file contains {len(leakage_list)}."
        )
    leakage_index_set = set(leakage_list)
    if contaminated:
        validate_contamination_metadata(config, leakage_indices, test_global_ids)

    model = build_model(config, data.x.size(1)).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])

    # Review thresholds are set on validation, then applied once to test.
    val_logits, val_labels = predict(model, data, data.val_mask, device)
    test_logits, test_labels = predict(model, data, data.test_mask, device)

    results = compute_metrics(test_logits, test_labels)
    if contaminated:
        results['evaluation_status'] = 'CONTAMINATED_SAME_LABEL_EVALUATION'
        results['valid_generalization_estimate'] = False
        results['checkpoint_selection_metric'] = 'contaminated_test_pr_auc'
        results['checkpoint_selection_warning'] = (
            'The same test split was used for training and contaminated checkpoint selection.')
        results['contamination_warning'] = ('All evaluated test labels were included in training. '
                                            'This PR-AUC is not a valid measure of generalization.')
    results['graph'] = graph_name
    results['budgets'] = budget_report(val_logits, test_logits, test_labels, args.budgets)

    # Separate the test nodes whose labels were deliberately exposed during
    # training from the remaining test nodes. Test predictions are returned in
    # ascending global node order because test_mask is a boolean mask.
    exposed_mask = torch.tensor(
        [int(node_id) in leakage_index_set for node_id in test_global_ids.tolist()],
        dtype=torch.bool
    )
    results['leakage_experiment'] = {
        'enabled': bool(len(leakage_indices)),
        'configured_test_fraction': configured_fraction,
        'exposed_test_nodes': int(exposed_mask.sum()),
        'total_test_nodes': int(len(test_global_ids)),
        'exposed_fraction_actual': float(exposed_mask.float().mean()) if len(exposed_mask) else 0.0,
        'exposed_subset_metrics': compute_metrics(test_logits[exposed_mask], test_labels[exposed_mask])
            if exposed_mask.any() else None,
        'unexposed_subset_metrics': compute_metrics(test_logits[~exposed_mask], test_labels[~exposed_mask])
            if (~exposed_mask).any() else None,
        'warning': ('Leakage-experiment scores are contaminated because selected test labels '
                    'were used for training; do not report them as unseen-data generalization.')
    }

    # Diagnostics. Validation numbers are recomputed here so they can be checked
    # against the value logged during training. They do not affect the test numbers.
    results['val'] = compute_metrics(val_logits, val_labels)
    results['best_training_epoch'] = best_training_epoch(args.exp_dir)

    print("Validation (recomputed):")
    print(f"  PR-AUC:  {results['val']['pr_auc']:.4f}")
    print(f"  ROC-AUC: {results['val']['roc_auc']:.4f}")
    if results['best_training_epoch']:
        b = results['best_training_epoch']
        print(f"  best training epoch {b['epoch']} of {b['epochs_logged']} logged, "
              f"logged val PR-AUC {b['logged_val_pr_auc']:.4f}")

    if contaminated:
        print("CONTAMINATED SAME-LABEL EVALUATION RESULTS")
    print("Test results:")
    print(f"  Average Precision (AP; JSON field pr_auc): {results['pr_auc']:.4f}")
    print(f"  ROC-AUC: {results['roc_auc']:.4f}")
    leak = results['leakage_experiment']
    print(f"Leakage experiment: {leak['exposed_test_nodes']}/{leak['total_test_nodes']} "
          f"test labels exposed ({leak['exposed_fraction_actual']:.1%})")
    if leak['exposed_subset_metrics'] is not None:
        print(f"  Exposed-subset PR-AUC:   {leak['exposed_subset_metrics']['pr_auc']:.4f}")
    if leak['unexposed_subset_metrics'] is not None:
        print(f"  Unexposed-subset PR-AUC: {leak['unexposed_subset_metrics']['pr_auc']:.4f}")
    for row in results['budgets']:
        print(f"  budget {row['budget']:.0%}: precision {row['precision_at_budget']:.4f}, "
              f"recall {row['recall_at_budget']:.4f}, flagged {row['flagged_fraction']:.2%}")

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    # Per-row test scores, with the same row order as the graph test mask.
    np.savez(scores_path, scores=test_logits.numpy(), labels=test_labels.numpy(),
             global_node_ids=test_global_ids.numpy(), leakage_exposed=exposed_mask.numpy())
    print(f"Saved {out_path} and {scores_path}")


if __name__ == "__main__":
    main()
