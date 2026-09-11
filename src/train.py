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

def parse_args():
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
    return parser.parse_args()

def load_config(args):
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    config = vars(args).copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            config.update(yaml_config)
    
    config.update(cli_args)
    
    if config.get('run_name') is None:
        config['run_name'] = f"{config['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return config

def main():
    args = parse_args()
    config = load_config(args)
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
    best_val_pr_auc = 0.0
    metrics_log = []

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = 0
        total_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index).squeeze(-1)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size].float())

            if is_camouflage_model:
                # Supervise the camouflage layers' internal fraud-score
                # heads with the SAME batch's true labels, over ALL sampled
                # nodes in this batch (not just the seed nodes), since the
                # score head's job is to inform neighbor trust for every
                # node that participates in message passing, not just the
                # ones being scored for the final prediction.
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

        if val_metrics['pr_auc'] > best_val_pr_auc:
            best_val_pr_auc = val_metrics['pr_auc']
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pt'))
            print("  --> Saved new best model")

    # Save metrics
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_log, f, indent=4)
    print("Training finished.")

if __name__ == "__main__":
    main()