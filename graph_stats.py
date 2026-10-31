"""Print exact partition fraud counts and graph sizes for the processed graph."""
import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", choices=["causal", "full"], default="causal")
    args = parser.parse_args()
    path = os.path.join("data", "processed", f"graph_{args.graph}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. First run: python src/data/build_graph.py"
        )
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Graph file: {path}")
    print(f"Graph mode: {args.graph}")
    print(f"Total nodes/transactions: {data.num_nodes:,}")
    print(f"Total directed edges: {data.edge_index.size(1):,}")
    print(f"Feature columns: {data.x.size(1):,}")
    print()
    for name in ("train", "val", "test"):
        mask = getattr(data, f"{name}_mask")
        labels = data.y[mask]
        total = int(mask.sum())
        fraud = int((labels == 1).sum())
        legit = int((labels == 0).sum())
        rate = (fraud / total * 100) if total else 0
        print(f"{name.upper():5s}: total={total:,} | fraud={fraud:,} | "
              f"legitimate={legit:,} | fraud_rate={rate:.3f}%")
    print()
    print("Note: counts reflect the exact locally built graph and its chronological split.")


if __name__ == "__main__":
    main()
