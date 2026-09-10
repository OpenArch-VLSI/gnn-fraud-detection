import sys
import os
import torch
import torch.nn as nn

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, proj_root)

from src.models.layers.sage import GraphSAGEModel

def test_sage():
    num_nodes = 5
    in_channels = 16
    hidden_channels = 32

    x = torch.randn((num_nodes, in_channels), requires_grad=True)

    # Ring among nodes 0-3 only; node 4 is deliberately isolated (zero in-degree)
    # to exercise the self-loop / zero-degree handling in SAGEConv
    edge_index = torch.tensor([
        [0, 1, 2, 3, 1, 2, 3, 0],
        [1, 2, 3, 0, 0, 1, 2, 3]
    ], dtype=torch.long)

    y = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0]])

    model = GraphSAGEModel(in_channels, hidden_channels)

    logits = model(x, edge_index)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    assert logits.shape == (num_nodes, 1), f"Expected shape {(num_nodes, 1)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Output contains NaN — zero-degree node not handled correctly"

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, y)
    loss.backward()

    assert x.grad is not None, "Input gradients are None"
    assert torch.sum(torch.abs(x.grad)) > 0, "Input gradients are all zero"

    # Node 4 is isolated — if the self-loop fix were missing or broken,
    # this gradient could be zero (dead node) rather than a real value
    assert torch.sum(torch.abs(x.grad[4])) > 0, "Isolated node's gradient is zero — self-loop fix may be broken"

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert torch.sum(torch.abs(param.grad)) > 0, f"Gradient for {name} is all zero"

    print("GraphSAGE synthetic graph test passed successfully!")
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradients computed successfully.")

if __name__ == "__main__":
    test_sage()