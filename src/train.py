import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from datetime import datetime

# Adjust path so we can run from anywhere
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.models import FraudGAT, FraudCamouflageGNN
from src.models.layers.sage import GraphSAGEModel
from src.utils.metrics import compute_metrics
from src.utils.seed import set_seed
from torch_geometric.loader import NeighborLoader

def build_parser():
    parser = argparse.ArgumentParser(description="Train Fraud Detection GNN")
    parser.add_argument('--config', type=str, help="Path to config YAML file")
    parser.add_argument('--model', type=str, choices=['sage', 'gat', 'camouflage'], default='sage')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--patience', type=int, default=10,
                         help="Stop training early if validation PR-AUC "
                              "hasn't improved for this many consecutive "
                              "epochs. Set to 0 to disable early stopping "
                              "and always run the full --epochs count.")
    parser.add_argument('--resume_dir', type=str, default=None,
                         help="Path to a mounted previous exp_dir (e.g. "
                              "/kaggle/input/<slug>/experiments/<run_name>) "
                              "containing last_model.pt, metrics.json, and "
                              "config.json to resume from. If omitted, "
                              "starts training from scratch.")
    return parser

def parse_args():
    return build_parser().parse_args()

def load_config(args):
    # BUGFIX: the previous version applied YAML values first, then
    # unconditionally overwrote them with argparse's parsed namespace
    # (config.update(cli_args)) -- but argparse fills in *default* values
    # for every flag the user didn't pass, so cli_args was never just "what
    # the user explicitly typed". That meant config.update(cli_args) wiped
    # out YAML-set fields like `model`, `lr`, etc. back to their argparse
    # defaults on every run, even when --config was the only flag given.
    #
    # Fix: only let CLI args override the YAML when the user actually
    # passed that flag on the command line (i.e. it differs from
    # argparse's own default for that arg), not merely because argparse
    # populated it with a default value.
    parser_defaults = {action.dest: action.default
                        for action in build_parser()._actions
                        if action.dest != 'help'}

    config = parser_defaults.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            config.update(yaml_config)

    # Now only apply CLI args that differ from the parser's own default --
    # these are the ones the user actually explicitly passed.
    explicit_cli_args = {
        k: v for k, v in vars(args).items()
        if k in parser_defaults and v != parser_defaults[k]
    }
    config.update(explicit_cli_args)

    if config.get('run_name') is None:
        config['run_name'] = f"{config['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # DEBUG: print the resolved config and exp_dir so we can confirm
    # exactly what this run is using, given the bug we're diagnosing.
    print("=== DEBUG: resolved config ===")
    print(json.dumps(config, indent=2))
    print(f"=== DEBUG: exp_dir will be: {os.path.join('experiments', config['run_name'])} ===")

    return config

def validate_resume_config(old_config: dict, new_config: dict) -> None:
    """
    Raises ValueError if new_config's model-defining fields don't match
    old_config's. Without this check, resuming with a mismatched --model,
    --hidden_channels, or --heads/--dropout (for gat/camouflage) would make
    load_state_dict either error out with a confusing shape-mismatch, or in
    rarer cases silently misload weights into the wrong-shaped tensors.
    """
    # Fields that affect model architecture for every model type.
    always_checked = ['model', 'hidden_channels']
    # Fields that only apply to gat/camouflage -- sage's GraphSAGEModel
    # constructor (see models.py / train.py's instantiation branch) only
    # takes (in_channels, hidden_channels), no heads/dropout.
    conditional_checked = ['heads', 'dropout']

    mismatches = []
    for key in always_checked:
        if old_config.get(key) != new_config.get(key):
            mismatches.append(
                f"  {key}: old={old_config.get(key)!r} vs new={new_config.get(key)!r}")

    if new_config['model'] in ('gat', 'camouflage'):
        for key in conditional_checked:
            if old_config.get(key) != new_config.get(key):
                mismatches.append(
                    f"  {key}: old={old_config.get(key)!r} vs new={new_config.get(key)!r}")

    if mismatches:
        raise ValueError(
            "Cannot resume: the following config fields differ between the "
            "checkpoint being resumed and this run's config, which would "
            "make load_state_dict error or silently misload:\n"
            + "\n".join(mismatches)
        )

def main():
    args = parse_args()
    config = load_config(args)

    # If resuming, load the old run's config up front so we can validate
    # architecture compatibility before doing any real work (data loading,
    # model construction, etc.) -- fail fast rather than partway through.
    old_config = None
    if config.get('resume_dir'):
        old_config_path = os.path.join(config['resume_dir'], 'config.json')
        if not os.path.exists(old_config_path):
            raise FileNotFoundError(
                f"--resume_dir given as {config['resume_dir']} but no "
                f"config.json found there -- check the mounted input path.")
        with open(old_config_path, 'r') as f:
            old_config = json.load(f)
        validate_resume_config(old_config, config)

    set_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create experiment dir
    exp_dir = os.path.join('experiments', config['run_name'])
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Load data
    data_path = os.path.join('data', 'processed', 'graph.pt')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Graph data not found at {data_path}. Run build_graph.py first.")
    
    # NOTE: Keep `data` on CPU here. NeighborLoader samples mini-batches from
    # this object on the fly; moving the *entire* graph (45.5M edges + 590K
    # feature rows) to GPU up front defeats the purpose of mini-batch/
    # neighbor-sampled training and risks OOM on a T4. Only the sampled
    # `batch` (already moved to `device` inside the train/val loops below)
    # needs to live on GPU.
    data = torch.load(data_path, weights_only=False)
    print(f"Loaded graph with {data.num_nodes} nodes and {data.edge_index.size(1)} edges.")

    in_channels = data.x.size(1)

    # Instantiate model
    if config['model'] == 'sage':
        model = GraphSAGEModel(in_channels, config['hidden_channels']).to(device)
    elif config['model'] == 'gat':
        model = FraudGAT(in_channels, config['hidden_channels'], heads=config['heads'], dropout=config['dropout']).to(device)
    elif config['model'] == 'camouflage':
        model = FraudCamouflageGNN(in_channels, config['hidden_channels'], heads=config['heads'], dropout=config['dropout']).to(device)
    else:
        raise ValueError("Unknown model type")

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=5e-4)

    # Calculate class weights for BCE (handling class imbalance)
    train_labels = data.y[data.train_mask]
    num_pos = train_labels.sum().item()
    num_neg = len(train_labels) - num_pos
    pos_weight = torch.tensor([num_neg / (num_pos + 1e-6)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # For the camouflage-resistant model only: an auxiliary BCE loss trains
    # each CamouflageGATConv layer's internal `score_head` to predict true
    # fraud labels. Without this, the label-aware trust signal the layer
    # relies on (see camouflage_gat.py) would be computed from an untrained,
    # meaningless score head -- this is what actually makes it label-aware
    # rather than just an unsupervised, arbitrarily-initialized detour.
    # Weighted low relative to the main loss since it's a supporting signal,
    # not the primary training objective.
    aux_loss_weight = config.get('aux_loss_weight', 0.3)
    is_camouflage_model = (config['model'] == 'camouflage')

    print("Setting up data loaders...")
    if os.environ.get('EVAL_PLUMBING_TEST') == '1':
        print("Warning: EVAL_PLUMBING_TEST=1. Using dummy loader for plumbing tests.")
        from torch_geometric.data import Data
        dummy_batch = Data(x=torch.randn(2, in_channels).to(device), 
                           edge_index=torch.tensor([[0, 1], [1, 0]]).to(device), 
                           y=torch.tensor([0, 1]).to(device), 
                           batch_size=2)
        train_loader = [dummy_batch]
        val_loader = [dummy_batch]
    else:
        train_loader = NeighborLoader(
            data,
            num_neighbors=[25, 10],
            batch_size=1024,
            input_nodes=data.train_mask,
            shuffle=True,
            num_workers=0,
        )
        
        val_loader = NeighborLoader(
            data,
            num_neighbors=[25, 10],
            batch_size=2048,
            input_nodes=data.val_mask,
            shuffle=False,
            num_workers=0,
        )

    print("Starting training...")

    # Resume state (overridden below if --resume_dir was given). Kept as a
    # single block so there's exactly one place these get initialized --
    # no duplicate/conflicting initializations later in the function.
    start_epoch = 1
    best_val_pr_auc = 0.0
    metrics_log = []
    # Tracks how many consecutive epochs have passed since val PR-AUC last
    # improved. Reset to 0 every time a new best is found; if it reaches
    # config['patience'], training stops early instead of running the full
    # --epochs count. This avoids burning GPU hours on epochs that aren't
    # actually helping once the model has plateaued.
    epochs_without_improvement = 0
    patience = config.get('patience', 10)

    if config.get('resume_dir'):
        checkpoint_path = os.path.join(config['resume_dir'], 'last_model.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"--resume_dir given as {config['resume_dir']} but no "
                f"last_model.pt found there.")
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_pr_auc = checkpoint.get('best_val_pr_auc', 0.0)
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)

        # Reload prior epoch-by-epoch history so the new version's
        # metrics.json continues the full record instead of restarting it,
        # which would otherwise silently drop everything from the previous
        # commit's run.
        old_metrics_path = os.path.join(config['resume_dir'], 'metrics.json')
        if os.path.exists(old_metrics_path):
            with open(old_metrics_path, 'r') as f:
                metrics_log = json.load(f)

        print(f"Resuming from epoch {start_epoch} "
              f"(best_val_pr_auc so far: {best_val_pr_auc:.4f}, "
              f"epochs_without_improvement: {epochs_without_improvement})")

    for epoch in range(start_epoch, config['epochs'] + 1):
        model.train()
        total_loss = 0
        total_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index).squeeze(-1)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size].float())

            if is_camouflage_model:
                node_scores = model.auxiliary_node_scores()
                aux_loss = criterion(node_scores, batch.y.float())
                loss = loss + aux_loss_weight * aux_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
            
        avg_train_loss = total_loss / total_batches

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index).squeeze(-1)
                loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size].float())
                val_loss += loss.item()
                val_batches += 1
                all_preds.append(out[:batch.batch_size])
                all_labels.append(batch.y[:batch.batch_size])
                
        avg_val_loss = val_loss / max(1, val_batches)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        val_metrics = compute_metrics(all_preds, all_labels)

        log_str = (f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | "
                   f"Val Loss: {avg_val_loss:.4f} | "
                   f"Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
                   f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        print(log_str)

        metrics_log.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            **val_metrics
        })

        # Rewrite metrics.json every epoch (not just after the loop ends) so
        # that a mid-run kill (session timeout, commit time limit, crash)
        # still leaves the full epoch-by-epoch history on disk up to the
        # last completed epoch, instead of losing it all.
        with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_log, f, indent=4)

        if val_metrics['pr_auc'] > best_val_pr_auc:
            best_val_pr_auc = val_metrics['pr_auc']
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_pr_auc': best_val_pr_auc,
            }, os.path.join(exp_dir, 'best_model.pt'))
            print("  --> Saved new best model")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Rolling "last epoch" checkpoint, saved every epoch regardless of
        # whether it's a new best. Includes optimizer state, the epoch
        # number, and early-stopping counters (not just model weights) so a
        # killed run can be *resumed* from exactly where it left off,
        # rather than only being usable for evaluation/fine-tuning the way
        # a bare state_dict would be. Saved after the improvement check
        # above so these counters reflect this epoch's outcome, not the
        # previous epoch's.
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_pr_auc': best_val_pr_auc,
            'epochs_without_improvement': epochs_without_improvement,
        }, os.path.join(exp_dir, 'last_model.pt'))

        if epochs_without_improvement > 0:
            # patience == 0 means early stopping is disabled entirely --
            # always run the full requested number of epochs.
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"  --> No val PR-AUC improvement for {patience} "
                      f"consecutive epochs (best so far: {best_val_pr_auc:.4f} "
                      f"at an earlier epoch). Stopping early at epoch {epoch}/"
                      f"{config['epochs']}.")
                break

    # Save metrics
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_log, f, indent=4)
    print("Training finished.")

if __name__ == "__main__":
    main()