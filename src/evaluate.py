import os
import argparse
import json
import torch
import sys

# Adjust path so we can run from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.models import FraudGAT, FraudCamouflageGNN
from src.models.layers.sage import GraphSAGEModel
from src.utils.metrics import compute_metrics
from torch_geometric.loader import NeighborLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fraud Detection GNN")
    parser.add_argument('--exp_dir', type=str, required=True, help="Path to experiment directory containing best_model.pt and config.json")
    return parser.parse_args()

def main():
    args = parse_args()
    
    config_path = os.path.join(args.exp_dir, 'config.json')
    model_path = os.path.join(args.exp_dir, 'best_model.pt')
    
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Experiment directory must contain config.json and best_model.pt")
        
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_path = os.path.join('data', 'processed', 'graph.pt')
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False).to(device)
    print("Data loaded successfully.")
    in_channels = data.x.size(1)

    # Instantiate model
    if config['model'] == 'sage':
        model = GraphSAGEModel(in_channels, config.get('hidden_channels', 64)).to(device)
    elif config['model'] == 'gat':
        model = FraudGAT(in_channels, config.get('hidden_channels', 64), heads=config.get('heads', 4), dropout=config.get('dropout', 0.3)).to(device)
    elif config['model'] == 'camouflage':
        model = FraudCamouflageGNN(in_channels, config.get('hidden_channels', 64), heads=config.get('heads', 4), dropout=config.get('dropout', 0.3)).to(device)
    else:
        raise ValueError("Unknown model type")

    # BUGFIX: train.py saves checkpoints as a dict with multiple keys
    # ({'model': state_dict, 'optimizer': ..., 'epoch': ..., ...}), not as
    # a bare state_dict. Passing that whole dict straight into
    # load_state_dict() raises a RuntimeError (missing/unexpected keys)
    # every time -- this was silently blocking all test-set evaluation for
    # every model, since run_experiments.py calls this script right after
    # every training run and would fail here immediately.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print("Setting up test data loader...")
    if os.environ.get('EVAL_PLUMBING_TEST') == '1':
        print("Warning: EVAL_PLUMBING_TEST=1. Using dummy loader for plumbing tests.")
        from torch_geometric.data import Data
        dummy_batch = Data(x=torch.randn(2, in_channels).to(device), 
                           edge_index=torch.tensor([[0, 1], [1, 0]]).to(device), 
                           y=torch.tensor([0, 1]).to(device), 
                           batch_size=2)
        test_loader = [dummy_batch]
    else:
        test_loader = NeighborLoader(
            data,
            num_neighbors=[25, 10],
            batch_size=2048,
            input_nodes=data.test_mask,
            shuffle=False,
            num_workers=0,
        )

    print("Evaluating on test set...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index).squeeze(-1)
            all_preds.append(out[:batch.batch_size])
            all_labels.append(batch.y[:batch.batch_size])
            
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    test_metrics = compute_metrics(all_preds, all_labels)

    print("Test Results:")
    print(f"  PR-AUC:  {test_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  F1-Score:{test_metrics['f1']:.4f}")

    # Save test metrics
    with open(os.path.join(args.exp_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)

if __name__ == "__main__":
    main()