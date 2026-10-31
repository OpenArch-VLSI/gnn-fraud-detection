import os
import argparse
import json
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch_geometric.loader import NeighborLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.models import FraudGAT, FraudCamouflageGNN, GraphSAGEModel
from src.utils.losses import pairwise_auc_loss, smooth_ap_loss
from src.utils.metrics import compute_metrics, selection_score
from src.utils.seed import set_seed

FANOUT = [25, 10]          # neighbours sampled per hop; evaluate.py uses the same
VAL_SAMPLE_SEED = 0        # fixed so validation scores are repeatable across epochs
SELECTION_METRIC = 'pr_auc'
GRAPH_CHOICES = ['causal', 'full']


def build_parser(suppress_defaults=False):
    """
    With suppress_defaults=True no defaults are set, so parse_args() returns
    only the flags the user actually typed. load_config() uses this to tell
    explicit flags apart from argparse's own defaults.
    """
    d = argparse.SUPPRESS if suppress_defaults else None
    parser = argparse.ArgumentParser(description="Train fraud detection GNN")
    parser.add_argument('--config', type=str, help="Path to config YAML file")
    parser.add_argument('--model', type=str, choices=['sage', 'gat', 'camouflage'],
                        default=d if suppress_defaults else 'sage')
    parser.add_argument('--graph', type=str, choices=GRAPH_CHOICES,
                        default=d if suppress_defaults else 'causal')
    parser.add_argument('--epochs', type=int, default=d if suppress_defaults else 100)
    parser.add_argument('--lr', type=float, default=d if suppress_defaults else 0.01)
    parser.add_argument('--hidden_channels', type=int, default=d if suppress_defaults else 64)
    parser.add_argument('--heads', type=int, default=d if suppress_defaults else 4)
    parser.add_argument('--dropout', type=float, default=d if suppress_defaults else 0.3)
    parser.add_argument('--seed', type=int, default=d if suppress_defaults else 42)
    parser.add_argument('--run_name', type=str, default=d if suppress_defaults else None)
    parser.add_argument('--patience', type=int, default=d if suppress_defaults else 10,
                        help="Stop if validation PR-AUC has not improved for this many epochs. "
                             "0 disables early stopping.")
    parser.add_argument('--rank_loss_weight', type=float, default=d if suppress_defaults else 1.0,
                        help="Weight of the pairwise ROC-AUC surrogate. 0 removes the term.")
    parser.add_argument('--ap_loss_weight', type=float, default=d if suppress_defaults else 1.0,
                        help="Weight of the smooth average-precision surrogate. 0 removes the term.")
    parser.add_argument('--ap_temperature', type=float, default=d if suppress_defaults else 1.0,
                        help="Sigmoid temperature of the smooth-AP surrogate, in logit units.")
    parser.add_argument('--resume_dir', type=str, default=d if suppress_defaults else None,
                        help="Previous exp_dir containing last_model.pt, metrics.json and config.json.")
    parser.add_argument('--leakage_test_fraction', type=float,
                        default=d if suppress_defaults else 0.0,
                        help=("Leakage experiment only: fraction of test nodes whose labels "
                              "are deliberately added to training seeds. Default 0.0."))
    return parser


def load_config(args):
    explicit = vars(build_parser(suppress_defaults=True).parse_args())
    explicit.pop('config', None)
    defaults = {a.dest: a.default for a in build_parser()._actions
                if a.dest not in ('help', 'config')}

    config = dict(defaults)
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config, 'r') as f:
            config.update(yaml.safe_load(f) or {})
    config.update(explicit)

    if config['graph'] not in GRAPH_CHOICES:
        raise ValueError(f"graph must be one of {GRAPH_CHOICES}, got {config['graph']!r}")
    leak_fraction = float(config.get('leakage_test_fraction', 0.0))
    if not 0.0 <= leak_fraction <= 1.0:
        raise ValueError("leakage_test_fraction must be between 0.0 and 1.0")
    config['leakage_test_fraction'] = leak_fraction
    if config.get('selection_metric', SELECTION_METRIC) != SELECTION_METRIC:
        raise ValueError(f"selection_metric must be '{SELECTION_METRIC}', "
                         f"got {config.get('selection_metric')!r}")
    config['selection_metric'] = SELECTION_METRIC

    if config.get('run_name') is None:
        config['run_name'] = f"{config['model']}_{config['graph']}_" \
                             f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return config


def validate_resume_config(old_config: dict, new_config: dict) -> None:
    """
    Refuses to resume if a model-defining or objective-defining field differs.
    Checkpoints written before a field existed have None for it and are refused.
    """
    always_checked = ['model', 'graph', 'hidden_channels', 'rank_loss_weight',
                      'ap_loss_weight', 'ap_temperature', 'selection_metric',
                      'leakage_test_fraction']
    conditional_checked = ['heads', 'dropout']

    keys = list(always_checked)
    if new_config['model'] in ('gat', 'camouflage'):
        keys += conditional_checked
    if new_config['model'] == 'camouflage':
        keys.append('detach_trust')

    mismatches = [f"  {k}: old={old_config.get(k)!r} vs new={new_config.get(k)!r}"
                  for k in keys if old_config.get(k) != new_config.get(k)]
    if mismatches:
        raise ValueError("Cannot resume: these config fields differ from the checkpoint:\n"
                         + "\n".join(mismatches))


def training_objective(criterion, seed_logits, seed_labels, config):
    """Weighted BCE on the seed nodes, plus optional pairwise-ROC and smooth-PR surrogates."""
    loss = criterion(seed_logits, seed_labels.float())
    if config['rank_loss_weight'] > 0:
        loss = loss + config['rank_loss_weight'] * pairwise_auc_loss(seed_logits, seed_labels)
    if config['ap_loss_weight'] > 0:
        loss = loss + config['ap_loss_weight'] * smooth_ap_loss(
            seed_logits, seed_labels, temperature=config['ap_temperature'])
    return loss


@torch.no_grad()
def validate(model, val_loader, criterion, config, device):
    """Validation with repeatable neighbour sampling. fork_rng leaves the training RNG untouched."""
    model.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(VAL_SAMPLE_SEED)
        val_loss, n_batches = 0.0, 0
        preds, labels = [], []
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index).squeeze(-1)
            seed_logits = out[:batch.batch_size]
            seed_labels = batch.y[:batch.batch_size]
            val_loss += training_objective(criterion, seed_logits, seed_labels, config).item()
            n_batches += 1
            preds.append(seed_logits.cpu())
            labels.append(seed_labels.cpu())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    return val_loss / max(1, n_batches), compute_metrics(preds, labels)


def main():
    args = build_parser().parse_args()
    config = load_config(args)

    if config.get('resume_dir'):
        old_config_path = os.path.join(config['resume_dir'], 'config.json')
        if not os.path.exists(old_config_path):
            raise FileNotFoundError(f"--resume_dir given but no config.json in {config['resume_dir']}")
        with open(old_config_path, 'r') as f:
            validate_resume_config(json.load(f), config)

    # Refuse to overwrite an existing experiment. This protects checkpoints,
    # metrics, and test results from accidental replacement on reruns.
    run_name = str(config['run_name'])
    if run_name in ('', '.', '..') or os.path.basename(run_name) != run_name:
        raise ValueError("run_name must be a simple folder name, not a path.")
    exp_dir = os.path.join('experiments', run_name)
    if os.path.isdir(exp_dir) and os.listdir(exp_dir):
        raise FileExistsError(
            f"Experiment directory already exists and is not empty: {exp_dir}. "
            "Choose a new --run_name. Existing experiment outputs are never "
            "overwritten automatically."
        )

    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    data_path = os.path.join('data', 'processed', f"graph_{config['graph']}.pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Graph not found at {data_path}. Run src/data/build_graph.py first.")
    # The graph stays on CPU; NeighborLoader moves sampled batches to the device.
    data = torch.load(data_path, weights_only=False)
    print(f"Loaded {data_path}: {data.num_nodes} nodes, {data.edge_index.size(1)} edges.")
    in_channels = data.x.size(1)

    if config['model'] == 'sage':
        model = GraphSAGEModel(in_channels, config['hidden_channels']).to(device)
    elif config['model'] == 'gat':
        model = FraudGAT(in_channels, config['hidden_channels'],
                         heads=config['heads'], dropout=config['dropout']).to(device)
    else:
        model = FraudCamouflageGNN(in_channels, config['hidden_channels'],
                                   heads=config['heads'], dropout=config['dropout'],
                                   detach_trust=config.get('detach_trust', True)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=5e-4)

    # Normal training nodes plus a reproducibly selected fraction of test nodes.
    # This intentionally leaks labels for a controlled experiment; the resulting
    # test PR-AUC is not an unbiased estimate of performance on unseen labels.
    training_mask = data.train_mask.clone()
    test_indices = torch.where(data.test_mask)[0]
    leak_fraction = float(config.get('leakage_test_fraction', 0.0))
    n_leak = int(round(leak_fraction * len(test_indices)))
    if n_leak > 0:
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(config['seed']) + 901)
        chosen = torch.randperm(len(test_indices), generator=generator)[:n_leak]
        leakage_indices = test_indices[chosen]
        training_mask[leakage_indices] = True
    else:
        leakage_indices = torch.empty(0, dtype=torch.long)

    torch.save(leakage_indices.cpu(),
               os.path.join(exp_dir, 'leakage_test_indices.pt'))
    train_labels = data.y[training_mask]
    num_pos = train_labels.sum().item()
    num_neg = len(train_labels) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    aux_loss_weight = config.get('aux_loss_weight', 0.3)
    is_camouflage_model = (config['model'] == 'camouflage')

    print(f"Leakage experiment: fraction={leak_fraction:.3f}; "
          f"test labels exposed to training={len(leakage_indices):,}/{len(test_indices):,}")
    print(f"Training seed nodes total: {int(training_mask.sum()):,}")

    train_loader = NeighborLoader(data, num_neighbors=FANOUT, batch_size=1024,
                                  input_nodes=training_mask, shuffle=True, num_workers=0)
    val_loader = NeighborLoader(data, num_neighbors=FANOUT, batch_size=2048,
                                input_nodes=data.val_mask, shuffle=False, num_workers=0)

    start_epoch = 1
    best_val_score = 0.0
    metrics_log = []
    epochs_without_improvement = 0
    patience = config.get('patience', 10)

    if config.get('resume_dir'):
        checkpoint_path = os.path.join(config['resume_dir'], 'last_model.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No last_model.pt in {config['resume_dir']}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_score = checkpoint.get('best_val_score', 0.0)
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        old_metrics_path = os.path.join(config['resume_dir'], 'metrics.json')
        if os.path.exists(old_metrics_path):
            with open(old_metrics_path, 'r') as f:
                metrics_log = json.load(f)
        print(f"Resumed at epoch {start_epoch} (best validation PR-AUC so far: {best_val_score:.4f})")

    print("Starting training...")
    for epoch in range(start_epoch, config['epochs'] + 1):
        model.train()
        total_loss, total_batches = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index).squeeze(-1)
            seed_logits = out[:batch.batch_size]
            seed_labels = batch.y[:batch.batch_size]
            loss = training_objective(criterion, seed_logits, seed_labels, config)

            if is_camouflage_model:
                # Only train seed nodes supervise score_head, so no val/test labels reach it.
                node_scores = model.auxiliary_node_scores()[:batch.batch_size]
                loss = loss + aux_loss_weight * criterion(node_scores, seed_labels.float())

            loss.backward()
            # GAT's exponential attention can spike on heavy-tailed inputs; clipping limits the damage.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        avg_train_loss = total_loss / max(1, total_batches)

        avg_val_loss, val_metrics = validate(model, val_loader, criterion, config, device)
        val_score = selection_score(val_metrics)

        print(f"Epoch {epoch:03d} | Train loss {avg_train_loss:.4f} | Val loss {avg_val_loss:.4f} | "
              f"Val PR-AUC {val_metrics['pr_auc']:.4f} | Val ROC-AUC {val_metrics['roc_auc']:.4f}")

        metrics_log.append({'epoch': epoch, 'train_loss': avg_train_loss,
                            'val_loss': avg_val_loss, 'val_score': val_score, **val_metrics})
        with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_log, f, indent=4)

        if val_score > best_val_score:
            best_val_score = val_score
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'best_val_score': best_val_score},
                       os.path.join(exp_dir, 'best_model.pt'))
            print("  --> saved new best model")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'best_val_score': best_val_score,
                    'epochs_without_improvement': epochs_without_improvement},
                   os.path.join(exp_dir, 'last_model.pt'))

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"  --> no validation PR-AUC improvement for {patience} epochs; "
                  f"stopping at epoch {epoch}/{config['epochs']} (best {best_val_score:.4f}).")
            break

    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_log, f, indent=4)
    print("Training finished.")


if __name__ == "__main__":
    main()